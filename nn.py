from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np


class EncoderLayer(nn.Module):
    num_heads: int = 8
    embed_dim: int = 128
    ff_dim: int = 512
    bn_epsilon: float = 1e-5

    def setup(self):

        self.mha = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.embed_dim,
            out_features=self.embed_dim,
        )

        self.mlp1 = nn.Dense(self.ff_dim)
        self.mlp2 = nn.Dense(self.embed_dim)

        self.bn1 = nn.BatchNorm(epsilon=self.bn_epsilon)
        self.bn2 = nn.BatchNorm(epsilon=self.bn_epsilon)

    def __call__(self, x, mask=None, training=True):
        # Self attention
        mha_out = self.mha(x, x, mask=mask, deterministic=not training)
        bn1_out = self.bn1(x + mha_out, use_running_average=not training)

        # MLP
        mlp_out = self.mlp1(bn1_out)
        mlp_out = nn.relu(mlp_out)
        mlp_out = self.mlp2(mlp_out)
        bn2_out = self.bn2(x + mlp_out, use_running_average=not training)

        return bn2_out


class GraphTransformerEncoder(nn.Module):
    embed_dim: int = 128
    num_heads: int = 8
    num_layers: int = 3
    ff_dim: int = 512

    def setup(self):
        self.encode_depot = nn.Dense(self.embed_dim)
        self.encode_customers = nn.Dense(self.embed_dim)
        self.encoder_layers = [
            EncoderLayer(
                num_heads=self.num_heads,
                embed_dim=self.embed_dim,
                ff_dim=self.ff_dim,
            )
            for _ in range(self.num_layers)
        ]

    def __call__(self, x, mask=None, training=True):
        coords, demands = x

        x = jnp.concatenate(
            [
                self.encode_depot(coords[:, 0, None, :]),
                self.encode_customers(
                    jnp.concatenate([coords[:, 1:, :], demands[:, 1:, None]], axis=-1)
                ),
            ],
            axis=1,
        )

        for layer in self.encoder_layers:
            x = layer(x, mask=mask, training=training)

        return x


class PointerDecoder(nn.Module):
    embed_dim: int = 128
    num_heads: int = 8
    clip: float = 10.0

    def setup(self):
        self.Wq_fixed = nn.Dense(self.embed_dim, use_bias=False)
        self.Wq_step = nn.Dense(self.embed_dim, use_bias=False)
        self.Wk = nn.Dense(self.embed_dim, use_bias=False)
        self.mha = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.embed_dim,
            out_features=self.embed_dim,
        )

    def __call__(self, node_embeddings, step_context, mask, training=True):

        Q_step = self.Wq_step(step_context)
        Q_fixed = self.Wq_fixed(jnp.sum(node_embeddings, axis=1)[:, None])

        Q1 = Q_fixed + Q_step
        Q2 = self.mha(
            Q1,
            node_embeddings,
            mask=jnp.transpose(mask[:, None], (0, 1, 3, 2)),
        )
        K2 = self.Wk(node_embeddings)

        logits = jnp.einsum("...qd, ...kd -> ...qk", Q2, K2) / jnp.sqrt(self.embed_dim)
        logits = self.clip * nn.tanh(logits)

        logits = jnp.where(
            jnp.transpose(mask, (0, 2, 1)),
            jnp.ones_like(logits) * -np.inf,
            logits,
        )

        return logits[:, 0]


class AttentionModel(nn.Module):
    embed_dim: int = 256
    num_heads: int = 8
    num_layers: int = 3
    ff_dim: int = 1024
    clip: float = 10.0

    def setup(self):
        self.encoder = GraphTransformerEncoder(
            self.embed_dim,
            self.num_heads,
            self.num_layers,
            self.ff_dim,
        )
        self.decoder = PointerDecoder(
            self.embed_dim,
            self.num_heads,
            self.clip,
        )

    def encode(self, x, training=True):
        return self.encoder(x, training=training)

    def decode(self, x, node_embeddings, mask=None, training=True):
        return self.decoder(x, node_embeddings, mask=mask, training=training)


@partial(jax.jit, static_argnums=(0,), static_argnames=("deterministic", "training"))
def solve(net, variables, rng, problems, deterministic=False, training=True):
    coords, demands = problems

    encoder_variables, decoder_variables = variables

    bs = coords.shape[0]
    seq_len = coords.shape[1]

    encoder_state, _ = encoder_variables.pop("params")

    node_embeddings, encoder_state = net.apply(
        encoder_variables,
        problems,
        mutable=encoder_state.keys(),
        method=net.encode,
        training=training,
    )

    # Create masks for each node type.
    customer_mask = jnp.ones((bs, seq_len), dtype=np.bool).at[:, 0].set(0)

    # Initialize visited mask with only the depot.
    visited = jnp.zeros((bs, seq_len), dtype=jnp.int32).at[:, 0].set(1)

    # Initialize the capacities as one.
    capacities = jnp.ones(bs)

    # Initialize sequences with 2 * max_nodes.
    # This represents the worst case for a VRP where each node visit is
    # followed by a depot visit.
    sequences = jnp.zeros((bs, 2 * seq_len), dtype=jnp.int32)

    step_context = jnp.concatenate(
        [node_embeddings[:, 0, None], capacities[:, None, None]], axis=-1
    )

    log_probs = jnp.zeros(bs)

    def decode_iteration(idx, loop_state):
        (rng, sequences, visited, capacities, step_context, log_probs) = loop_state

        current_node = sequences[:, idx - 1]

        # Build the next node mask.
        next_node_mask = (
            # We cannot visit already visited customers.
            (visited & customer_mask)
            # We cannot visit nodes exceeding the vehicle capacity.
            | (demands > capacities[:, None])
        )

        # We disallow the vehicle to stay in the node.
        next_node_mask = next_node_mask.at[jnp.arange(bs), current_node].set(True)

        # We always allow the vehicle to move or stay in a depot if all
        # customers have been visited.
        # FIXME: Zero as depot is hardcoded here.
        next_node_mask = next_node_mask.at[:, 0].set(
            ~(visited | ~customer_mask).all(axis=-1) & next_node_mask[:, 0],
        )

        decoder_state, _ = decoder_variables.pop("params")

        logits, decoder_state = net.apply(
            decoder_variables,
            node_embeddings,
            step_context,
            next_node_mask[:, :, None],
            mutable=decoder_state.keys(),
            method=net.decode,
            training=training,
        )

        if deterministic:
            next_nodes = jnp.argmax(logits, axis=-1)

        else:
            probs = nn.softmax(logits, axis=-1)
            rng, sample_rng = jax.random.split(rng)
            rngs = jax.random.split(sample_rng, bs)
            next_nodes = jax.vmap(partial(jax.random.choice, a=seq_len))(
                key=rngs, p=probs
            )

        sequences = sequences.at[jnp.arange(bs), idx].set(next_nodes)
        visited = visited.at[jnp.arange(bs), next_nodes].set(1)
        capacities = capacities - demands[jnp.arange(bs), next_nodes]
        capacities = jnp.where(next_nodes == 0, 1.0, capacities)

        prev_node_embedding = node_embeddings[jnp.arange(bs), next_nodes]
        step_context = jnp.concatenate(
            [prev_node_embedding[:, None], capacities[:, None, None]], axis=-1
        )

        # Store the log-probability of the selected node (for training).
        step_log_probs = nn.log_softmax(logits, axis=-1)[jnp.arange(bs), next_nodes]
        log_probs = log_probs + step_log_probs

        return (rng, sequences, visited, capacities, step_context, log_probs)

    init = (rng, sequences, visited, capacities, step_context, log_probs)
    _, sequences, _, _, _, log_probs = jax.lax.fori_loop(
        1, 2 * seq_len, decode_iteration, init
    )

    # state = init
    # for idx in range(1, max_len):
    #     state = decode_iteration(idx, state)

    # _, sequences, _, _, _, log_probs = state

    cost = get_costs(coords, sequences)

    return cost, log_probs, sequences


def get_cost(coords, route):
    max_size = coords.shape[0]

    src = jnp.tile(coords[None], (max_size, 1, 1))
    tgt = jnp.tile(coords[:, None], (1, max_size, 1))

    distances = tgt - src
    distances = jnp.sum(distances ** 2, axis=-1) ** 0.5

    src = route[:-1]
    dst = route[1:]

    step_distances = distances[src, dst]

    return jnp.sum(step_distances)


def get_costs(coords, routes):
    return jax.vmap(get_cost)(coords, routes)

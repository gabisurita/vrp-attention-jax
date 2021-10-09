from dataclasses import dataclass
from functools import partial
from typing import Callable, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from data import VRP


class TransformerEncoderBlock(nn.Module):
    num_heads: int = 8
    embed_dim: int = 128
    ff_dim: int = 512
    dropout_rate: float = 0.1

    def setup(self):
        self.mha = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.embed_dim,
            out_features=self.embed_dim,
            dropout_rate=self.dropout_rate,
        )

        self.mlp1 = nn.Dense(self.ff_dim)
        self.mlp2 = nn.Dense(self.embed_dim)
        # Kool et al. doesn't use dropout, but it usually helps.
        self.dropout = nn.Dropout(self.dropout_rate)

        # Paper uses BatchNorm instead of LayerNorm, but BatchNorm is usually
        # unsuitable for variable length sequences.
        # We go with the standard NLP LayerNorm style.
        # TODO: consider ViT style normalization, with LayerNorm before MHA.
        # https://stats.stackexchange.com/questions/474440/
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()

    def __call__(self, x, mask=None, deterministic=False):
        # Self attention
        mha_out = self.mha(x, x, mask=mask, deterministic=deterministic)
        mha_out = self.norm1(x + mha_out)

        # MLP
        mlp_out = self.mlp1(mha_out)
        mlp_out = nn.relu(mlp_out)
        mlp_out = self.dropout(mlp_out, deterministic=deterministic)
        mlp_out = self.mlp2(mlp_out)
        out = self.norm2(mha_out + mlp_out)

        return out


class VRPNodeEmbedding(nn.Module):
    embed_dim: int = 128

    def setup(self):
        self.encode_depot = nn.Dense(self.embed_dim)
        self.encode_customers = nn.Dense(self.embed_dim)

    def __call__(self, vrp):
        return jnp.concatenate(
            [
                self.encode_depot(vrp.coords[..., :1, :]),
                self.encode_customers(
                    jnp.concatenate(
                        [vrp.coords[..., 1:, :], vrp.demands[..., 1:, None]],
                        axis=-1,
                    )
                ),
            ],
            axis=-2,
        )


class VRPEncoder(nn.Module):
    embed_dim: int = 128
    num_heads: int = 8
    num_layers: int = 3
    ff_dim: int = 512

    def setup(self):
        self.embedding = VRPNodeEmbedding(self.embed_dim)
        self.encoder_blocks = [
            TransformerEncoderBlock(
                num_heads=self.num_heads,
                embed_dim=self.embed_dim,
                ff_dim=self.ff_dim,
            )
            for _ in range(self.num_layers)
        ]

    def __call__(self, vrp, deterministic=False):
        x = self.embedding(vrp)

        # Build 3D attention mask from 2D self-attention mask.
        mask = nn.make_attention_mask(vrp.mask > 0, vrp.mask > 0)

        for block in self.encoder_blocks:
            x = block(x, mask=mask, deterministic=deterministic)

        return x


class PointerDecoder(nn.Module):
    embed_dim: int = 128
    num_heads: int = 8
    clip: float = 10.0

    def setup(self):
        self.encode_state = nn.Dense(self.embed_dim)
        self.encode_graph = nn.Dense(self.embed_dim)
        self.encode_capacity = nn.Dense(self.embed_dim)

        self.Wk = nn.Dense(self.embed_dim)
        self.Wq = nn.Dense(self.embed_dim)

        self.mha = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.embed_dim,
            out_features=self.embed_dim,
        )

    def __call__(self, node_embeddings, states, mask=None, deterministic=False):
        state_nodes, capacities = states
        bs = node_embeddings.shape[0]

        # This part is a little different than the original paper. Instead of concatenating
        # then projecting, we project first, and then add instead of concatenate.
        # This is a more BERTasque approach to create the embedding.
        state_node_embedding = self.encode_state(
            node_embeddings[jnp.arange(bs), state_nodes]
        )
        graph_embedding = self.encode_graph(jnp.sum(node_embeddings, axis=1))
        capacity_embedding = self.encode_capacity(capacities[:, None])

        Q1 = (state_node_embedding + graph_embedding + capacity_embedding)[:, None]
        Q2 = self.mha(
            Q1,
            node_embeddings,
            mask=mask[:, None, None, :],
        )

        Q2 = self.Wq(Q2)
        K2 = self.Wk(node_embeddings)

        logits = jnp.einsum("...qd, ...kd -> ...qk", Q2, K2) / jnp.sqrt(self.embed_dim)
        logits = self.clip * nn.tanh(logits)

        logits = jnp.where(
            mask[:, None, :],
            jnp.ones_like(logits) * -np.inf,
            logits,
        )

        return logits[:, 0]


@dataclass(unsafe_hash=True)
class AttentionModel:
    embed_dim: int = 128
    num_heads: int = 8
    num_layers: int = 3
    ff_dim: int = 512
    clip: float = 10.0

    def __post_init__(self):
        self.encoder = VRPEncoder(
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

    def init(self, rng):
        # Init values, used only to mimic the problem shapes.
        b = 1
        n = 1

        mask = jnp.zeros((b, n))
        coords = jnp.zeros((b, n, 2))
        demands = jnp.zeros((b, n))
        vrp = VRP(mask, coords, demands)

        e = jnp.zeros((b, n, self.embed_dim))
        s = jnp.zeros(b, dtype=jnp.int32)
        c = jnp.zeros(b)
        m = jnp.zeros((b, n), dtype=np.bool)

        encoder_variables = self.encoder.init({"params": rng, "dropout": rng}, vrp)
        decoder_variables = self.decoder.init(
            {"params": rng, "dropout": rng}, e, (s, c), m
        )

        params = (encoder_variables["params"], decoder_variables["params"])
        return params

    @partial(jax.jit, static_argnums=(0,), static_argnames=("deterministic",))
    def solve(self, params, rng, problems, deterministic=False):

        encoder_params, decoder_params = params

        def encode(rng, problems):
            node_embeddings = self.encoder.apply(
                {"params": encoder_params},
                problems,
                rngs={"dropout": rng},
                deterministic=deterministic,
            )
            return node_embeddings

        def decode(rng, encoded, states, mask):
            logits = self.decoder.apply(
                {"params": decoder_params},
                encoded,
                states,
                mask,
                deterministic=deterministic,
                rngs={"dropout": rng},
            )
            return logits

        # if deterministic:

        #     routes, log_probs = jax.vmap(partial(
        #         decode_beam_search,
        #         encode,
        #         decode,
        #         rng,
        #         deterministic=deterministic,
        #     ))(problem=problems)

        #     cost_beams = lambda problem, routes: jax.vmap(partial(get_cost, problem))(routes)
        #     costs = jax.vmap(cost_beams)(problems.coords, routes)

        #     best = costs.argmin(axis=-1)
        #     bs = routes.shape[0]
        #     costs = costs[jnp.arange(bs), best]
        #     routes = routes[jnp.arange(bs), best]
        #     log_probs = log_probs[jnp.arange(bs), best]

        # else:
        routes, log_probs = decode_greedy(
            encode,
            decode,
            rng,
            problems,
            deterministic=deterministic,
        )
        costs = get_costs(problems.coords, routes)

        return costs, log_probs, routes


VRPEncoded = jnp.ndarray

VRPDecoderState = Tuple[jnp.array, jnp.array]

NodeMask = jnp.array

EncoderFn = Callable[[jax.random.PRNGKey, VRP], VRPEncoded]

DecoderFn = Callable[
    [jax.random.PRNGKey, VRPEncoded, VRPDecoderState, NodeMask], jnp.array
]


@partial(jax.jit, static_argnums=(0, 1), static_argnames=("deterministic",))
def decode_greedy(encoder, decoder, rng, problem, deterministic=False):

    bs = problem.coords.shape[0]
    seq_len = problem.coords.shape[1]

    # Call encoder.
    rng, encoder_rng = jax.random.split(rng)
    encoded = encoder(encoder_rng, problem)

    # Create masks for each node type.
    invalid_mask = problem.mask == False
    customer_mask = problem.mask.at[:, 0].set(0)

    # Initialize visited mask with only the depot.
    visited = jnp.zeros((bs, seq_len), dtype=jnp.int32).at[:, 0].set(1)

    # Initialize the capacities as one.
    capacities = jnp.ones(bs)

    # Initialize sequences with 2 * max_nodes.
    # This represents the worst case for a VRP where each node visit is
    # followed by a depot visit.
    sequences = jnp.zeros((bs, 2 * seq_len), dtype=jnp.int32)

    # Store the sum of log probs.
    log_probs = jnp.zeros(bs)

    def decode_iteration(idx, loop_state):
        (rng, sequences, visited, capacities, log_probs) = loop_state

        current_node = sequences[:, idx - 1]

        # Build the next node mask.
        next_node_mask = (
            invalid_mask
            # We cannot visit already visited customers.
            | (visited & customer_mask)
            # We cannot visit nodes exceeding the vehicle capacity.
            | (problem.demands > capacities[:, None])
        )

        # We disallow the vehicle to stay in the node.
        next_node_mask = next_node_mask.at[jnp.arange(bs), current_node].set(True)

        # We always allow the vehicle to move or stay in a depot if all
        # customers have been visited.
        next_node_mask = next_node_mask.at[:, 0].set(
            ~(visited | ~customer_mask).all(axis=-1) & next_node_mask[:, 0],
        )

        # Call decoder.
        rng, decoder_step_rng = jax.random.split(rng)
        logits = decoder(
            decoder_step_rng,
            encoded,
            (current_node, capacities),
            next_node_mask,
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
        capacities = capacities - problem.demands[jnp.arange(bs), next_nodes]
        capacities = jnp.where(next_nodes == 0, 1.0, capacities)

        # Store the log-probability of the selected node (for training).
        step_log_probs = nn.log_softmax(logits, axis=-1)[jnp.arange(bs), next_nodes]
        log_probs = log_probs + step_log_probs

        return (rng, sequences, visited, capacities, log_probs)

    init = (rng, sequences, visited, capacities, log_probs)
    _, sequences, _, _, log_probs = jax.lax.fori_loop(
        1, 2 * seq_len, decode_iteration, init
    )

    return sequences, log_probs


@partial(jax.jit, static_argnums=(0, 1), static_argnames=("deterministic", "num_beams"))
def decode_beam_search(
    encoder, decoder, rng, problem, deterministic=False, num_beams=20
):

    max_nodes = problem.coords.shape[-2]

    # Call encoder.
    rng, encoder_rng = jax.random.split(rng)
    encoded = encoder(encoder_rng, problem)

    # Create masks for each node type.
    invalid_mask = problem.mask == False
    customer_mask = problem.mask.at[0].set(0)

    # Initialize sequences with 2 * max_nodes.
    # This represents the worst case for a VRP where each node visit is
    # followed by a depot visit.
    sequences = jnp.zeros((num_beams, 2 * max_nodes), dtype=jnp.int32)

    # Initialize visited mask with only the depot.
    visited = jnp.zeros((num_beams, max_nodes), dtype=jnp.int32)

    # FIXME: Zero as depot is hardcoded here.
    visited = jax.ops.index_update(visited, jax.ops.index[..., 0], 1)

    # Initialize the capacities as one.
    capacities = jnp.ones(num_beams)

    # Initialize only one beam as zero, and others with a very negative
    # log prob. Without this, we would only repeat the decoded sequence.
    sequence_log_probs = -jnp.inf * jnp.ones(num_beams)
    sequence_log_probs = jax.ops.index_update(sequence_log_probs, 0, 0)

    # Declares the initial state of the decoding procedure.
    init = (rng, sequences, visited, capacities, sequence_log_probs)

    def beam_search_iteration(idx, loop_state):
        """
        Describes one iteration of the beam search procedure.
        """

        (rng, sequences, visited, capacities, sequence_log_probs) = loop_state

        # Represents the current node index.
        current_node = sequences[:, idx - 1]

        # Build the next node mask.
        next_node_mask = (
            invalid_mask
            # We cannot visit already visited customers.
            | (visited & customer_mask)
            # We cannot visit nodes exceeding the vehicle capacity.
            | (problem.demands[None] > capacities[:, None])
        )

        # We disallow the vehicle to stay in the node.
        next_node_mask = next_node_mask.at[jnp.arange(num_beams), current_node].set(
            True
        )

        # We always allow the vehicle to move or stay in a depot if all
        # customers have been visited.
        next_node_mask = next_node_mask.at[:, 0].set(
            ~(visited | ~customer_mask).all(axis=-1) & next_node_mask[:, 0],
        )

        # Call decoder.
        rng, decoder_step_rng = jax.random.split(rng)

        logits = decoder(
            decoder_step_rng,
            jnp.tile(encoded[None, :, :], (num_beams, 1, 1)),
            (current_node, capacities),
            next_node_mask,
        )

        log_probs = nn.log_softmax(logits, axis=-1)

        # Compute the transition log probs assigning an infinite negative
        # logprob to to forbidden nodes.
        transition_log_probs = sequence_log_probs[:, None] + log_probs

        # Compute top k transitions over the flattened array.
        values, indices = jax.lax.top_k(transition_log_probs.flatten(), num_beams)

        # Retrive beam indexes and node indexes from the flattened array.
        beam_indices = indices // max_nodes
        next_nodes = indices - beam_indices * max_nodes
        next_sequence_log_probs = values

        sequences = sequences[beam_indices].at[..., idx].set(next_nodes)
        visited = visited[beam_indices].at[jnp.arange(num_beams), next_nodes].set(1)
        capacities = capacities[beam_indices] - problem.demands[..., next_nodes]
        capacities = jnp.where(next_nodes == 0, 1.0, capacities)

        return (
            rng,
            sequences,
            visited,
            capacities,
            next_sequence_log_probs,
        )

    _, sequences, _, _, sequence_log_probs = jax.lax.fori_loop(
        1, 2 * max_nodes, beam_search_iteration, init
    )

    return sequences, sequence_log_probs


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

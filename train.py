import pickle
from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import jax.tools.colab_tpu
import mlflow
import numpy as np
import optax
import tensorflow as tf
import flax
from flax.training.train_state import TrainState
from scipy.stats import ttest_rel
from simple_parsing import ArgumentParser
from tqdm.auto import tqdm

from data import ProblemConfig, create_dataset
from nn import AttentionModel

# Disable GPUs and TPUs for TensorFlow, as we only use it for data loading.
tf.config.set_visible_devices([], "GPU")
tf.config.set_visible_devices([], "TPU")


@dataclass
class TrainConfig:

    task: str = ""
    """Task name, used to identify the run."""

    seed: int = 0
    """Random global seed."""

    max_customers: int = 50
    """Maximum number customers in the problem. Doesn't include the depot."""

    min_customers: int = 50
    """Minimum number customers in the problem. Doesn't include the depot."""

    num_train_samples: int = 128_000
    """Number of training samples per epoch."""

    num_validation_samples: int = 10_000
    """Number of validation samples per epoch."""

    capacity: int = 40
    """Capacity of the vehicle."""

    batch_size: int = 512
    """Training batch size."""

    epochs: int = 1000
    """Number of training epochs."""

    warmup_epochs: int = 1
    """Number of warmup epochs (without rollout baseline)."""

    learning_rate: float = 1e-4
    """Adam learning rate."""

    gradient_clipping: float = 1.0
    """Global gradient clipping."""

    weight_decay: float = 1e-3
    """AdamW weight decay."""

    save_path: str = "./weights/"
    """Path to save model checkpoints."""

    load_path: Optional[str] = None
    """Load a previous checkpoint."""

    use_tqdm: bool = True
    """If false, disables tqdm."""

    log_every: int = 10
    """Frequency to log batch statistics."""


@jax.jit
def reinforce_step(state, rng, problems):
    def loss_fn(params):
        costs, log_probs, _ = state.apply_fn(params, rng, problems)
        baseline = jnp.mean(costs)
        loss = jnp.mean((costs - baseline) * log_probs)
        return loss, jnp.mean(costs)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, costs), grads = grad_fn(state.params)

    # Support gradient across devices.
    grads = jax.lax.pmean(grads, "batch")

    training_state = state.apply_gradients(grads=grads)
    return training_state, loss, costs


@jax.jit
def reinforce_with_rollout_step(state, baseline_params, rng, problems):
    def loss_fn(params):
        costs, log_probs, _ = state.apply_fn(params, rng, problems)
        baseline, _, _ = state.apply_fn(
            baseline_params, rng, problems, deterministic=True
        )
        loss = jnp.mean((costs - baseline) * log_probs)
        return loss, jnp.mean(costs)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, costs), grads = grad_fn(state.params)

    # Support gradient across devices.
    grads = jax.lax.pmean(grads, "batch")

    state = state.apply_gradients(grads=grads)
    return state, loss, costs


@jax.jit
def eval_step(state, params, rng, problems):
    costs, _, _ = state.apply_fn(params, rng, problems, deterministic=True)
    return costs


if __name__ == "__main__":

    # Setup colab TPUs, or just ignore.
    try:
        jax.tools.colab_tpu.setup_tpu()
    except KeyError:
        print(f"Failed to setup TPUs, COLAB_TPU_ADDR not set.")
        pass

    # Load train config.
    parser = ArgumentParser()
    parser.add_arguments(TrainConfig, dest="train_config")
    parser.add_arguments(AttentionModel, dest="model")
    args = parser.parse_args()

    cfg = args.train_config

    devices = jax.devices()

    print(devices)

    # assert cfg.num_train_samples % (
    #     cfg.batch_size * len(devices)
    # ), "Invalid sample x batch x device shape"

    dataset_config = ProblemConfig(
        num_samples=cfg.num_train_samples,
        min_customers=cfg.min_customers,
        max_customers=cfg.max_customers,
        capacity=cfg.capacity,
    )

    eval_dataset_config = ProblemConfig(
        num_samples=cfg.num_validation_samples,
        min_customers=cfg.min_customers,
        max_customers=cfg.max_customers,
        capacity=cfg.capacity,
    )

    model = args.model

    # Setup experiment logging.
    mlflow.set_experiment("/vrp-attention")
    mlflow.start_run()
    mlflow.log_params(cfg.__dict__)
    mlflow.log_params(model.__dict__)

    # Initialize global RNG.
    rng = jax.random.PRNGKey(cfg.seed)

    rng, train_data_rng = jax.random.split(rng)
    rng, val_data_rng = jax.random.split(rng)
    rng, params_rng = jax.random.split(rng)
    rng, rollout_rng = jax.random.split(rng)

    params = model.init(params_rng)

    # If a checkpoint is passed, preload weights from it.
    if cfg.load_path:
        with open(cfg.load_path, "rb") as f:
            params = pickle.load(f)

    parallel_reinforce_step = jax.pmap(
        reinforce_step,
        axis_name="batch",
    )

    parallel_reinforce_with_rollout_step = jax.pmap(
        reinforce_with_rollout_step,
        axis_name="batch",
    )

    parallel_eval_step = jax.pmap(
        eval_step,
        axis_name="batch",
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.gradient_clipping),
        # Use adam with weight decay as it sometimes leads to better generalization.
        optax.adamw(learning_rate=cfg.learning_rate, weight_decay=cfg.weight_decay),
    )

    # Create the training state with the model.solve funtion.
    state = TrainState.create(apply_fn=model.solve, params=params, tx=optimizer)

    # Replicate state across devices.
    state = flax.jax_utils.replicate(state)

    # Copy params to the baseline model.
    baseline_params = state.params

    # Sample a random evaluation dataset.
    val_data_rng, val_epoch_rng = jax.random.split(val_data_rng)
    eval_dataset = create_dataset(eval_dataset_config, val_epoch_rng)

    # Training loop.
    for epoch in range(cfg.epochs):

        # Sample the training data.
        train_data_rng, train_epoch_rng = jax.random.split(train_data_rng)
        train_dataset = (
            create_dataset(dataset_config, train_epoch_rng)
            .batch(cfg.batch_size // len(devices))
            .batch(len(devices))
            .prefetch(tf.data.AUTOTUNE)
            .enumerate()
            .as_numpy_iterator()
        )

        batches_per_epoch = int(np.ceil(cfg.num_train_samples // cfg.batch_size))
        pbar = tqdm(train_dataset, total=batches_per_epoch, disable=not cfg.use_tqdm)

        for t, inputs in pbar:
            rollout_rng, step_rng = jax.random.split(rollout_rng)

            device_rngs = jax.random.split(rng, len(devices))

            # Warmup.
            if cfg.load_path is None and epoch < cfg.warmup_epochs:
                state, loss, costs = parallel_reinforce_step(state, device_rngs, inputs)

            # Greedy rollout baseline.
            else:
                state, loss, costs = parallel_reinforce_with_rollout_step(
                    state, baseline_params, device_rngs, inputs
                )

            # Means across devices.
            loss, costs = loss.mean(), costs.mean()

            step_metrics = {
                "epoch": epoch,
                "timestep": t,
                "train_loss": float(loss),
                "train_costs": float(costs),
            }

            pbar.set_postfix(loss=loss, costs=costs)

            if t % cfg.log_every == 0:
                mlflow.log_metrics(step_metrics)

        candidate_vals = jnp.empty((0,))
        baseline_vals = jnp.empty((0,))

        # Evaluation dataset.
        for problems in (
            eval_dataset.batch(cfg.batch_size // len(devices))
            .batch(len(devices))
            .as_numpy_iterator()
        ):
            device_rngs = jax.random.split(rng, len(devices))

            step_candidate_vals = parallel_eval_step(
                state, state.params, device_rngs, problems
            )
            step_baseline_vals = parallel_eval_step(
                state, baseline_params, device_rngs, problems
            )

            candidate_vals = jnp.concatenate(
                [candidate_vals, step_candidate_vals.flatten()]
            )
            baseline_vals = jnp.concatenate(
                [baseline_vals, step_baseline_vals.flatten()]
            )

        candidate_mean = jnp.mean(candidate_vals)
        baseline_mean = jnp.mean(baseline_vals)

        # Check if the candidate mean value is better.
        candidate_better = candidate_mean < baseline_mean

        # Check if improvement is significant with a one-sided t-test between related samples.
        _, p = ttest_rel(np.array(candidate_vals), np.array(baseline_vals))
        one_sided_p = p / 2

        statistically_significant = (one_sided_p / 2) < 0.05

        metrics = {
            "epoch": epoch,
            "validation_costs": float(candidate_mean),
            "baseline_costs": float(baseline_mean),
            "p-value": one_sided_p,
        }

        print(metrics)
        mlflow.log_metrics(metrics)

        # If model is statiscally better than the baseline, copy model parameters.
        if candidate_better and statistically_significant:

            baseline_params = state.params
            val_data_rng, val_epoch_rng = jax.random.split(val_data_rng)
            eval_dataset = create_dataset(eval_dataset_config, val_epoch_rng)

        # Save the model checkpoint.
        params = flax.jax_utils.unreplicate(state).params

        with open(f"{cfg.save_path}{cfg.task}_epoch{epoch}.pkl", "wb") as f:
            pickle.dump(params, f)

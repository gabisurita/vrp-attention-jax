import pickle
from dataclasses import dataclass
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
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

# Disable GPUs and TPUs for TensorFlow, as we only use it
# for data loading.
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

    capacity: int = 50
    """Capacity of the vehicle."""

    batch_size: int = 512
    """Training batch size."""

    epochs: int = 1000
    """Number of training epochs."""

    warmup_epochs: int = 1
    """Number of warmup epochs (without rollout baseline)."""

    learning_rate: float = 1e-4
    """Adam learning rate."""

    save_path: str = "./weights/"
    """Path to save model checkpoints."""

    load_path: Optional[str] = None
    """Load a previous checkpoint."""

    use_tqdm: bool = True
    """If false, disables tqdm."""

    log_every: int = 10
    """Frequency to log batch statistics."""


if __name__ == "__main__":

    # Load train config.
    parser = ArgumentParser()
    parser.add_arguments(TrainConfig, dest="train_config")
    parser.add_arguments(AttentionModel, dest="model")
    args = parser.parse_args()

    cfg = args.train_config

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

    devices = jax.devices()

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

    baseline_params = params

    @jax.jit
    def reinforce_step(rng, training_state, problems):
        def loss_fn(params):
            costs, log_probs, _ = model.solve(params, rng, problems)
            baseline = jnp.mean(costs)
            loss = jnp.mean((costs - baseline) * log_probs)
            return loss, jnp.mean(costs)

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, costs), grads = grad_fn(training_state.params)

        # Support gradient across devices.
        grads = jax.lax.pmean(grads, "problems")

        training_state = training_state.apply_gradients(grads=grads)
        return training_state, loss, costs

    @jax.jit
    def reinforce_with_rollout_step(rng, training_state, baseline_params, problems):
        def loss_fn(params):
            costs, log_probs, _ = model.solve(params, rng, problems)
            baseline, _, _ = model.solve(
                baseline_params, rng, problems, deterministic=True
            )
            loss = jnp.mean((costs - baseline) * log_probs)
            return loss, jnp.mean(costs)

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, costs), grads = grad_fn(training_state.params)

        # Support gradient across devices.
        grads = jax.lax.pmean(grads, "problems")

        training_state = training_state.apply_gradients(grads=grads)
        return training_state, loss, costs

    @jax.jit
    def parallel_reinforce_step(rng, training_state, problems):
        rngs = jax.random.split(rng, len(devices))

        states, loss, costs = jax.pmap(
            reinforce_step,
            axis_name="problems",
            donate_argnums=(0, 1),
        )(rngs, training_state, problems)

        loss, costs = loss.mean(), costs.mean()

        return states, loss, costs

    @jax.jit
    def parallel_reinforce_with_rollout_step(
        rng, training_state, baseline_params, problems
    ):
        rngs = jax.random.split(rng, len(devices))

        states, loss, costs = jax.pmap(
            reinforce_with_rollout_step,
            axis_name="problems",
            donate_argnums=(0, 1, 2),
        )(rngs, training_state, baseline_params, problems)

        loss, costs = loss.mean(), costs.mean()

        return states, loss, costs

    @jax.jit
    def eval_step(rng, params, problems):
        costs, _, _ = model.solve(params, rng, problems, deterministic=True)
        return costs

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        # Use adam with weight decay as it sometimes leads to better generalization.
        optax.adamw(learning_rate=cfg.learning_rate, weight_decay=1e-5),
    )

    training_state = TrainState.create(
        apply_fn=model.solve, params=params, tx=optimizer
    )

    # Sample a random evaluation dataset.
    val_data_rng, val_epoch_rng = jax.random.split(val_data_rng)
    eval_dataset = create_dataset(eval_dataset_config, val_epoch_rng)

    # Training loop.
    for epoch in range(cfg.epochs):

        parallel_baseline = flax.jax_utils.replicate(baseline_params)
        parallel_training_state = flax.jax_utils.replicate(training_state)

        # Sample the training data.
        train_data_rng, train_epoch_rng = jax.random.split(train_data_rng)
        dataset = (
            create_dataset(dataset_config, train_epoch_rng)
            .batch(cfg.batch_size // len(devices))
            .batch(len(devices), drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
            .enumerate()
            .as_numpy_iterator()
        )

        batches_per_epoch = int(np.ceil(cfg.num_train_samples // cfg.batch_size))
        pbar = tqdm(dataset, total=batches_per_epoch, disable=not cfg.use_tqdm)

        for t, inputs in pbar:
            rollout_rng, step_rng = jax.random.split(rollout_rng)

            # Warmup.
            if cfg.load_path is None and epoch < cfg.warmup_epochs:
                parallel_training_state, loss, costs = parallel_reinforce_step(
                    step_rng, parallel_training_state, inputs
                )

            # Greedy rollout baseline.
            else:
                (
                    parallel_training_state,
                    loss,
                    costs,
                ) = parallel_reinforce_with_rollout_step(
                    step_rng, parallel_training_state, parallel_baseline, inputs
                )

            step_metrics = {
                "epoch": epoch,
                "timestep": t,
                "train_loss": float(loss),
                "train_costs": float(costs),
            }

            pbar.set_postfix(loss=loss, costs=costs)

            if t % cfg.log_every == 0:
                mlflow.log_metrics(step_metrics)

        training_state = flax.jax_utils.unreplicate(parallel_training_state)

        # Evaluate candidate.
        candidate_vals = jnp.concatenate(
            [
                eval_step(rng, training_state.params, problems)
                for problems in eval_dataset.batch(cfg.batch_size).as_numpy_iterator()
            ]
        )

        baseline_vals = jnp.concatenate(
            [
                eval_step(rng, baseline_params, problems)
                for problems in eval_dataset.batch(cfg.batch_size).as_numpy_iterator()
            ]
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
            baseline_params = training_state.params
            val_data_rng, val_epoch_rng = jax.random.split(val_data_rng)
            eval_dataset = create_dataset(eval_dataset_config, val_epoch_rng)

        # Save the model checkpoint.
        with open(f"{cfg.save_path}{cfg.task}_epoch{epoch}.pkl", "wb") as f:
            pickle.dump(training_state.params, f)

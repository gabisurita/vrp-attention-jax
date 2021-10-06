import copy
import logging
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
from flax.training.train_state import TrainState
from scipy.stats import ttest_rel
from simple_parsing import ArgumentParser
from tqdm.auto import tqdm

from data import generate_data
from nn import AttentionModel, decode_greedy, get_costs

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

    num_nodes: int = 100
    """Number of nodes in the problem. Includes the depot."""

    num_samples: int = 128_000
    """Number of training samples per epoch."""

    num_validation_samples: int = 10_000
    """Number of validation samples per epoch."""

    batch_size: int = 256
    """Training batch_size size."""

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
    model = args.model

    # Setup experiment logging.
    mlflow.set_experiment("/vrp-attention")
    mlflow.start_run()
    mlflow.log_params(cfg.__dict__)
    mlflow.log_params(model.__dict__)

    # Initialize global RNG.
    rng = jax.random.PRNGKey(cfg.seed)
    params = model.init(rng)

    # If a checkpoint is passed, preload weights from it.
    if cfg.load_path:
        with open(cfg.load_path, "rb") as f:
            params = pickle.load(f)

    baseline_params = params

    @jax.jit
    def reinforce_with_rollout_loss(params, baseline_params, rng, problems):
        costs, log_probs, _ = model.solve(params, rng, problems)
        baseline, _, _ = model.solve(baseline_params, rng, problems, deterministic=True)
        loss = jnp.mean((costs - baseline) * log_probs)
        return loss, jnp.mean(costs)

    @jax.jit
    def reinforce_loss(params, _, rng, problems):
        costs, log_probs, _ = model.solve(params, rng, problems)
        baseline = jnp.mean(costs)
        loss = jnp.mean((costs - baseline) * log_probs)
        return loss, jnp.mean(costs)

    optimizer = optax.chain(
        optax.adam(learning_rate=cfg.learning_rate),
        optax.clip_by_global_norm(1.0),
    )

    training_state = TrainState.create(
        apply_fn=model.solve, params=params, tx=optimizer
    )

    # Sample a random evaluation dataset.
    eval_dataset = generate_data(cfg.num_validation_samples, cfg.num_nodes)

    for epoch in range(cfg.epochs):
        # Sample the training data.
        dataset = (
            generate_data(cfg.num_samples, cfg.num_nodes)
            .batch(cfg.batch_size)
            .prefetch(tf.data.AUTOTUNE)
            .enumerate()
            .as_numpy_iterator()
        )

        batches_per_epoch = int(np.ceil(cfg.num_samples / cfg.batch_size))
        pbar = tqdm(dataset, total=batches_per_epoch, disable=not cfg.use_tqdm)

        for t, inputs in pbar:
            rng, step_rng = jax.random.split(rng)

            # Warmup mean baseline.
            if cfg.load_path is None and epoch < cfg.warmup_epochs:
                loss_fn = reinforce_loss

            # Greedy rollout baseline.
            else:
                loss_fn = reinforce_with_rollout_loss

            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (loss, costs), grads = grad_fn(
                training_state.params, baseline_params, step_rng, inputs
            )
            training_state = training_state.apply_gradients(grads=grads)

            if t % cfg.log_every == 0:
                pbar.set_postfix(loss=loss, costs=costs)
                mlflow.log_metrics(
                    {
                        "epoch": epoch,
                        "timestep": t,
                        "train_loss": float(loss),
                        "train_costs": float(costs),
                    }
                )

        # Evaluate candidate.
        candidate_vals = jnp.concatenate(
            [
                model.solve(training_state.params, rng, problems, deterministic=True)[0]
                for problems in eval_dataset.batch(cfg.batch_size).as_numpy_iterator()
            ]
        )

        baseline_vals = jnp.concatenate(
            [
                model.solve(baseline_params, rng, problems, deterministic=True)[0]
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
            eval_dataset = generate_data(cfg.num_validation_samples, cfg.num_nodes)

        # Save the model checkpoint.
        with open(f"{cfg.save_path}{cfg.task}_epoch{epoch}.pkl", "wb") as f:
            pickle.dump(training_state.params, f)

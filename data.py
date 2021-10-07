from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import tensorflow as tf
from flax import struct

# Disable GPUs and TPUs for TensorFlow, as we only use it
# for data loading.
tf.config.set_visible_devices([], "GPU")
tf.config.set_visible_devices([], "TPU")


class VRP(NamedTuple):
    mask: jnp.ndarray
    coords: jnp.ndarray
    demands: jnp.ndarray


@struct.dataclass
class ProblemConfig:
    num_samples: int = 512
    min_customers: int = 50
    max_customers: int = 50
    min_demand: int = 1
    max_demand: int = 9
    capacity: int = 40


@partial(jax.jit, static_argnums=(0,))
def create_batch(config, rng):
    bs, n = config.num_samples, config.max_customers + 1

    rng, size_rng = jax.random.split(rng)
    sizes = jax.random.randint(
        size_rng,
        shape=(bs,),
        minval=1 + config.min_customers,
        maxval=config.max_customers,
    )

    mask = jnp.arange(n)[None, :] < sizes[:, None]

    rng, coords_rng = jax.random.split(rng)
    coords = jax.random.uniform(
        coords_rng,
        shape=(bs, n, 2),
        minval=0.0,
        maxval=1.0,
    )

    coords = jnp.where(mask[:, :, None], coords, 0.0)

    rng, demands_rng = jax.random.split(rng)
    demands = jax.random.randint(
        demands_rng,
        shape=(bs, n),
        minval=config.min_demand,
        maxval=config.max_demand,
    ) / jnp.array(config.capacity, dtype=jnp.float32)

    demands = jnp.where(mask, demands, 0.0).at[:, 0].set(0.0)

    return VRP(mask, coords, demands)


def create_dataset(config: ProblemConfig, rng: jax.random.PRNGKey) -> tf.data.Dataset:
    """Create a tensorflow dataset of VRPs with the given config."""
    batch = create_batch(config, rng)
    return tf.data.Dataset.from_tensor_slices(batch)

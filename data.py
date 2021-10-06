import tensorflow as tf

# Disable GPUs and TPUs for TensorFlow, as we only use it
# for data loading.
tf.config.set_visible_devices([], "GPU")
tf.config.set_visible_devices([], "TPU")


def generate_data(n_samples=1000, n_nodes=20, seed=None, min_size=0.01, max_size=0.2):
    if seed is None:
        g = tf.random.experimental.Generator.from_non_deterministic_state()
    else:
        g = tf.random.experimental.Generator.from_seed(seed)

    @tf.function
    def tf_rand():
        return [
            # Coords
            g.uniform(shape=[n_samples, n_nodes, 2], minval=0, maxval=1),
            # Demands
            tf.concat(
                [
                    tf.zeros((n_samples, 1)),
                    g.uniform(
                        shape=[n_samples, n_nodes - 1],
                        minval=min_size,
                        maxval=max_size,
                        dtype=tf.float32,
                    ),
                ],
                axis=-1,
            ),
        ]

    return tf.data.Dataset.from_tensor_slices(tuple(tf_rand()))

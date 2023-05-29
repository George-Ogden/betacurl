import tensorflow as tf
import numpy as np

def support_to_value(coefficients: tf.Tensor, support: tf.Tensor) -> tf.Tensor:
    """convert support (or supports) to value (or values)

    Args:
        coefficients (tf.Tensor): logits along the supports
        support (tf.Tensor): values to weight the support

    Returns:
        tf.Tensor: values
    """
    assert np.allclose(tf.reduce_sum(coefficients, axis=-1), 1), "coefficients must sum to 1"
    return tf.reduce_sum(
        support * coefficients,
        axis=-1
    )

def value_to_support(values: tf.Tensor, support: tf.Tensor) -> tf.Tensor:
    """convert values to supports

    Args:
        values (tf.Tensor): values to convert to support
        support (tf.Tensor): values to weight the support

    Returns:
        tf.Tensor: support
    """
    upper_bounds = tf.searchsorted(
        support,
        values,
        side="left"
    )
    lower_bounds = upper_bounds - 1
    # linear interpolate between lower and upper bound values
    interpolation = (
        values - tf.gather(support, lower_bounds)
    ) / (
        tf.gather(support, upper_bounds) - tf.gather(support, lower_bounds)
    )
    interpolation = interpolation[:, tf.newaxis]
    support = tf.one_hot(
        lower_bounds,
        depth=len(support),
        dtype=tf.float32
    ) * (1 - interpolation) + tf.one_hot(
        upper_bounds,
        depth=len(support),
        dtype=tf.float32
    ) * interpolation
    return support
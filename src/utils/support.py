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
    if support.ndim == 1:
        support = tf.tile(
            support[tf.newaxis, :],
            [coefficients.shape[0], 1]
        )
    assert support.ndim == 2, "support must be 1 or 2 dimensional"
    assert support.shape[-1] == coefficients.shape[-1], "support and coefficients must match"
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
    if support.ndim == 1:
        support = tf.tile(
            support[tf.newaxis, :],
            [values.shape[0], 1]
        )
    assert support.ndim == 2, "support must be 1 or 2 dimensional"
    assert values.ndim == 1, "values must be 1 dimensional"
    assert support.shape[-2] == values.shape[-1], "support and values must match"
    values = tf.reshape(
        values,
        (-1, 1)
    )
    upper_bounds = tf.searchsorted(
        support,
        values,
        side="left"
    )
    lower_bounds = upper_bounds - 1
    # additional indices for gathering
    additional_indices = tf.range(support.shape[0], dtype=tf.int32)
    lower_bounds = tf.stack(
        (
            additional_indices,
            tf.squeeze(
                lower_bounds,
                axis=-1
            )
        ),
        axis=-1
    )
    upper_bounds = tf.stack(
        (
            additional_indices,
            tf.squeeze(
                upper_bounds,
                axis=-1
            )
        ),
        axis=-1
    )
    #  linear interpolate between lower and upper bound values
    values = tf.squeeze(values, -1)
    interpolation = (
        values - tf.gather_nd(support, lower_bounds)
    ) / (
        tf.gather_nd(support, upper_bounds) - tf.gather_nd(support, lower_bounds)
    )
    lower_bounds = lower_bounds[:, 1]
    upper_bounds = upper_bounds[:, 1]
    interpolation = interpolation[:, tf.newaxis]
    support = tf.one_hot(
        lower_bounds,
        depth=support.shape[-1],
        dtype=tf.float32
    ) * (1 - interpolation) + tf.one_hot(
        upper_bounds,
        depth=support.shape[-1],
        dtype=tf.float32
    ) * interpolation
    return support
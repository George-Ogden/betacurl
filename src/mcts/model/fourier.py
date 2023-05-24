from tensorflow_probability import util
import tensorflow as tf
import numpy as np

from typing import Dict

from .comb import CombDistribution

class FourierDistribution(CombDistribution):
    granularity: int = 1000
    min_prob: float = 1e-2
    def __init__(
        self,
        coefficients: tf.Tensor,
        bounds: tf.Tensor,
        validate_args: bool = False,
    ):
        assert coefficients.shape[-1] == 2, "coefficients must come in pairs (sin-cos)"
        assert bounds.shape[-1] == 2, "the bounds must come in pairs (min-max)"
        self.batch_size = coefficients.shape[:-2]
        assert bounds.shape[:-1] == self.batch_size, "the bounds must have the same batch size as the coefficients"

        # store in original shape for reinitialisation
        self.coefficients = coefficients
        self.bounds = bounds

        # flatten for easier processing
        coefficients = tf.reshape(self.coefficients, (-1, *coefficients.shape[-2:]))
        bounds = tf.reshape(self.bounds, (-1, 2))

        # sample 1000 points
        sample_frequency = (
            2 * np.pi
            * tf.reshape(tf.linspace(0, 1, self.granularity), (self.granularity, 1))
            * tf.range(coefficients.shape[1], dtype=tf.float64)
        )

        # calculate amplitudes using the coefficients
        sample_amplitudes = tf.stack((tf.sin(sample_frequency), tf.cos(sample_frequency)), axis=-1)

        # calculate the pdf
        pdf = tf.maximum(
            tf.reduce_sum(
                tf.cast(
                    sample_amplitudes,
                    coefficients.dtype
                ) * tf.expand_dims(coefficients, 1),
                axis=(-1, -2)
            ),
            1e-3 # avoid numerical instability and almost zero probabilities
        )
        pdf /= tf.reduce_sum(pdf, axis=-1, keepdims=True)

        # reshape to original shape
        pdf = tf.reshape(pdf, (*self.batch_size, self.granularity))
        bounds = tf.reshape(bounds, (*self.batch_size, 2))

        super().__init__(
            pdf=pdf,
            bounds=bounds,
            validate_args=validate_args,
            name="FourierDistribution"
        )

    def _parameter_properties(self, dtype=None, num_classes=None) -> Dict[str, util.ParameterProperties]:
        return {
            "coefficients": util.ParameterProperties(),
            "bounds": util.ParameterProperties(),
        }
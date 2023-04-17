from __future__ import annotations

from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability import distributions, util
import tensorflow as tf
import numpy as np

from typing import Dict, Tuple

class FourierDistribution(distributions.Distribution):
    granularity: int = 1000
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

        dtype = dtype_util.common_dtype([coefficients, bounds], dtype_hint=tf.float32)
        super().__init__(
            dtype=dtype,
            reparameterization_type=distributions.FULLY_REPARAMETERIZED,
            validate_args=validate_args,
            allow_nan_stats=True,
            name="FourierDistribution",
        )

        # store in original shape for reinitialisation
        self.coefficients = coefficients
        self.bounds = bounds

        # flatten for easier processing
        self._coefficients = tf.reshape(self.coefficients, (-1, *coefficients.shape[-2:]))
        self._bounds = tf.reshape(self.bounds, (-1, 2))
        self.n = len(self._coefficients)

        # sample 1000 points
        sample_frequency = (
            2 * np.pi
            * tf.reshape(tf.linspace(0, 1, self.granularity), (self.granularity, 1))
            * tf.range(self._coefficients.shape[1], dtype=tf.float64)
        )

        # calculate amplitudes using the coefficients
        sample_amplitudes = tf.stack((tf.sin(sample_frequency), tf.cos(sample_frequency)), axis=-1)

        # calculate the cdf
        self.pdf = tf.maximum(
            tf.reduce_sum(
                tf.cast(
                    sample_amplitudes,
                    self.dtype
                ) * tf.expand_dims(self._coefficients, 1),
                axis=(-1, -2)
            ),
            1e-10 # avoid numerical instability
        )
        self.pdf /= tf.reduce_sum(self.pdf, axis=-1, keepdims=True)
        self.cdf = tf.concat((tf.cumsum(self.pdf, axis=-1), tf.maximum(1., tf.reduce_sum(self.pdf, axis=-1, keepdims=True))), axis=-1)
        self.points = tf.transpose(tf.linspace(self._bounds[:, 0], self._bounds[:, 1], self.granularity))
        assert self.points.shape == self.pdf.shape

    def _parameter_properties(self, dtype=None, num_classes=None) -> Dict[str, util.ParameterProperties]:
        return {
            "coefficients": util.ParameterProperties(),
            "bounds": util.ParameterProperties(),
        }

    def _sample_shape(self) -> Tuple[int, ...]:
        return ()

    def _batch_shape(self) -> Tuple[int, ...]:
        return self.batch_size

    def _sample_n(self, n: int, seed=None) -> tf.Tensor:
        # sample n points from the cdf using linear interpolation
        random_points = tf.random.uniform(
            (self.n, n),
            dtype=self.dtype,
            seed=seed
        )
        samples = tf.transpose(
            tf.gather_nd(
                self.points,
                tf.stack(
                    (
                        tf.stack(
                            [tf.range(self.n, dtype=tf.int32)] * n,
                            axis=-1
                        ),
                        tf.cast(
                            tf.searchsorted(
                                self.cdf,
                                random_points,
                                side="right"
                            ),
                            tf.int32
                        )
                    ),
                    axis=-1
                )
            )
        )
        return tf.reshape(
            samples,
            (n, *self.batch_size)
        )

    def _prob(self, action: tf.Tensor) -> tf.Tensor:
        action_shape = action.shape
        action = tf.reshape(
            tf.cast(action, self.dtype),
            (-1, np.prod(self.batch_shape))
        )

        # interpolate between pdf values
        scaled = (action - self._bounds[:, 0]) / (self._bounds[:, 1] - self._bounds[:, 0]) * (self.granularity - 1)
        pdf_indices = tf.minimum(tf.cast(scaled, tf.int32), self.granularity - 2)
        delta = scaled - tf.cast(pdf_indices, self.dtype)
        # additional index for gathering
        additional_indices = tf.stack(
            [tf.range(self.n, dtype=tf.int32)] * len(action),
            axis=0
        )
        return tf.reshape(
            tf.gather_nd(
                self.pdf,
                tf.stack(
                    (
                        additional_indices,
                        pdf_indices
                    ),
                    axis=-1
                )
            ) * delta + tf.gather_nd(
                self.pdf,
                tf.stack(
                    (
                        additional_indices,
                        pdf_indices + 1
                    ),
                    axis=-1
                )
            ) * (1 - delta),
            action_shape
        ) * self.granularity

    def _mean(self) -> tf.Tensor:
        return tf.reshape(
            tf.reduce_sum(self.points * self.pdf, axis=-1),
            self.batch_size
        )

    def _variance(self) -> tf.Tensor:
        return tf.reshape(
            tf.reduce_sum(self.points ** 2 * self.pdf, axis=-1),
            self.batch_size
        ) - self._mean() ** 2

    def _mode(self) -> tf.Tensor:
        return tf.reshape(
            tf.gather_nd(
                self.points,
                tf.stack(
                    (
                        tf.range(self.n),
                        tf.argmax(self.pdf, axis=-1, output_type=tf.int32)
                    ),
                    axis=-1
                )
            ),
            self.batch_size
        )

    def entropy(self) -> tf.Tensor:
        return tf.reshape(
            -tf.reduce_sum(self.pdf * tf.math.log(self.pdf), axis=-1) / self.granularity,
            self.batch_size
        )

    def kl_divergence(self, other: FourierDistribution) -> tf.Tensor:
        return tf.reshape(
            tf.reduce_sum(self.pdf * tf.math.log(self.pdf / other.pdf), axis=-1) / self.granularity,
            self.batch_size
        ) * self.granularity
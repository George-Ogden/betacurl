from __future__ import annotations

from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability import distributions, util
import tensorflow as tf
import numpy as np

from typing import Dict, Tuple

class CombDistribution(distributions.Distribution):
    def __init__(
        self,
        pdf: tf.Tensor,
        bounds: tf.Tensor,
        validate_args: bool = False,
        name: str = "CombDistribution",
    ):
        assert np.allclose(np.sum(pdf, axis=-1), 1.), "the pdf must sum to 1"
        assert bounds.shape[-1] == 2, "the bounds must come in pairs (min-max)"
        self.batch_size = pdf.shape[:-1]
        assert bounds.shape[:-1] == self.batch_size, "the bounds must have the same batch size as the pdf"

        dtype = dtype_util.common_dtype([pdf, bounds], dtype_hint=tf.float32)
        super().__init__(
            dtype=dtype,
            reparameterization_type=distributions.FULLY_REPARAMETERIZED,
            validate_args=validate_args,
            allow_nan_stats=True,
            name=name,
        )

        # store in original shape for reinitialisation
        self.bounds = bounds
        self.pdf = pdf

        # flatten for easier processing
        self._pdf = tf.reshape(pdf, (-1, pdf.shape[-1]))
        self._bounds = tf.reshape(self.bounds, (-1, 2))
        self.n = len(self._pdf)

        self._cdf = tf.concat((tf.cumsum(self._pdf, axis=-1), tf.maximum(1., tf.reduce_sum(self._pdf, axis=-1, keepdims=True))), axis=-1)
        self._points = tf.transpose(tf.linspace(self._bounds[:, 0], self._bounds[:, 1], self.granularity))
        assert self._points.shape == self._pdf.shape

    def _parameter_properties(self, dtype=None, num_classes=None) -> Dict[str, util.ParameterProperties]:
        return {
            "pdf": util.ParameterProperties(),
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
                self._points,
                tf.stack(
                    (
                        tf.stack(
                            [tf.range(self.n, dtype=tf.int32)] * n,
                            axis=-1
                        ),
                        tf.cast(
                            tf.searchsorted(
                                self._cdf,
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
                self._pdf,
                tf.stack(
                    (
                        additional_indices,
                        pdf_indices
                    ),
                    axis=-1
                )
            ) * delta + tf.gather_nd(
                self._pdf,
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
            tf.reduce_sum(self._points * self._pdf, axis=-1),
            self.batch_size
        )

    def _variance(self) -> tf.Tensor:
        return tf.reshape(
            tf.reduce_sum(self._points ** 2 * self._pdf, axis=-1),
            self.batch_size
        ) - self._mean() ** 2

    def _mode(self) -> tf.Tensor:
        return tf.reshape(
            tf.gather_nd(
                self._points,
                tf.stack(
                    (
                        tf.range(self.n),
                        tf.argmax(self._pdf, axis=-1, output_type=tf.int32)
                    ),
                    axis=-1
                )
            ),
            self.batch_size
        )

    def entropy(self) -> tf.Tensor:
        return tf.reshape(
            -tf.reduce_sum(self._pdf * tf.math.log(self._pdf), axis=-1) / self.granularity,
            self.batch_size
        )

    def kl_divergence(self, other: CombDistribution) -> tf.Tensor:
        return tf.reshape(
            tf.reduce_sum(self._pdf * tf.math.log(self._pdf / other.pdf), axis=-1) / self.granularity,
            self.batch_size
        ) * self.granularity
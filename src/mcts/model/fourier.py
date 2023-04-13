from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability import distributions
from tensorflow.keras import layers, losses
from tensorflow import data, keras
import tensorflow as tf
import numpy as np
import math

from typing import Callable, List, Optional, Tuple, Union

from ...model import DenseModelFactory, EmbeddingFactory, ModelFactory, TrainingConfig, BEST_MODEL_FACTORY
from ...game import GameSpec

from .config import FourierMCTSModelConfig
from .ppo import PPOMCTSModel

class FourierMCTSModel(PPOMCTSModel):
    ...

class FourierDistribution(distributions.Distribution):
    granularity: int = 1000
    def __init__(
        self,
        coefficients: tf.Tensor,
        range: tf.Tensor,
        validate_args: bool = False,
    ):
        assert coefficients.ndim == 3
        assert coefficients.shape[-1] == 2
        assert range.shape == (2,)
        dtype = dtype_util.common_dtype([coefficients], dtype_hint=tf.float32)
        super().__init__(
            dtype=dtype,
            reparameterization_type=distributions.FULLY_REPARAMETERIZED,
            validate_args=validate_args,
            allow_nan_stats=True,
            name="FourierDistribution",
        )
        # sample 1000 points
        sample_frequency = (
            2 * np.pi
            * tf.reshape(tf.linspace(0, 1, self.granularity), (self.granularity, 1))
            * tf.range(coefficients.shape[1], dtype=tf.float64)
        )

        # calculate amplitudes using the coefficients
        sample_amplitudes = tf.stack((tf.sin(sample_frequency), tf.cos(sample_frequency)), axis=-1)

        # calculate the cdf
        self.pdf = tf.maximum(
            tf.reduce_sum(
                tf.cast(
                    sample_amplitudes,
                    self.dtype
                ) * tf.expand_dims(coefficients, 1),
                axis=(-1, -2)
            ),
            0
        )
        self.pdf /= tf.reduce_sum(self.pdf, axis=-1, keepdims=True)
        self.cdf = tf.concat((tf.cumsum(self.pdf, axis=-1), tf.maximum(1., tf.reduce_sum(self.pdf, axis=-1, keepdims=True))), axis=-1)
        self.points = tf.linspace(*range, self.granularity)
        self.range = range

    def _sample_shape(self) -> Tuple[int, ...]:
        return ()

    def _batch_shape(self) -> Tuple[int, ...]:
        return self.pdf.shape[:-1]

    def _sample_n(self, n: int, seed=None) -> tf.Tensor:
        # sample n points from the cdf using linear interpolation
        random_points = tf.random.uniform(
            (len(self.pdf), n),
            dtype=self.dtype,
            seed=seed
        )
        return tf.transpose(
            tf.gather(
                self.points,
                tf.cast(
                    tf.searchsorted(
                        self.cdf,
                        random_points,
                        side="right"
                    ),
                    tf.int32
                )
            )
        )

    def _mean(self) -> tf.Tensor:
        return tf.reduce_sum(self.points * self.pdf, axis=-1)

    def _variance(self) -> tf.Tensor:
        return tf.reduce_sum(self.points ** 2 * self.pdf, axis=-1) - self._mean() ** 2

    def _mode(self) -> tf.Tensor:
        return tf.gather(self.points, tf.argmax(self.pdf, axis=-1))

    def prob(self, action: tf.Tensor) -> tf.Tensor:
        # interpolate between pdf values
        scaled = (action - self.range[0]) / (self.range[1] - self.range[0]) * (self.granularity - 1)
        indices = tf.minimum(tf.cast(scaled, tf.int32), self.granularity - 2)
        delta = scaled - tf.cast(indices, self.dtype)
        return tf.gather(self.pdf, indices, axis=-1) * delta + tf.gather(self.pdf, indices + 1, axis=-1) * (1 - delta)
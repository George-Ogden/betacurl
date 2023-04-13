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
        self.cdf = tf.cumsum(self.pdf, axis=-1)
        self.points = np.linspace(*range, self.granularity)
        self.range = range

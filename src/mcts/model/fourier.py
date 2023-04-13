from __future__ import annotations

from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability import distributions, util
import tensorflow as tf
import numpy as np

from typing import Dict, Optional, Tuple

from ...model import DenseModelFactory, ModelFactory, BEST_MODEL_FACTORY
from ...game import GameSpec

from .config import FourierMCTSModelConfig
from .ppo import PPOMCTSModel

class FourierMCTSModel(PPOMCTSModel):
    def __init__(
        self,
        game_spec: GameSpec,
        scaling_spec: Optional[np.ndarray] = None,
        model_factory: ModelFactory = BEST_MODEL_FACTORY,
        config: FourierMCTSModelConfig = FourierMCTSModelConfig()
    ):
        super().__init__(game_spec, scaling_spec, model_factory, config)
        assert self.action_shape == (1,)
        self.policy_head = DenseModelFactory.create_model(
            input_shape=self.feature_size,
            output_shape=(config.fourier_features, 2),
            config=DenseModelFactory.CONFIG_CLASS(
                output_activation="linear"
            )
        )

        self.setup_model()

    def _generate_distribution(self, raw_actions: tf.Tensor) -> distributions.Distribution:
        range = self.action_range.squeeze(-1)
        if raw_actions.ndim == 3:
            range = np.tile(self.action_range.squeeze(-1), (len(raw_actions), 1))
        return FourierDistribution(
            raw_actions,
            range=range
        )

class FourierDistribution(distributions.Distribution):
    granularity: int = 1000
    def __init__(
        self,
        coefficients: tf.Tensor,
        range: tf.Tensor,
        validate_args: bool = False,
    ):
        self.batched = coefficients.ndim != 2
        if not self.batched:
            coefficients = tf.expand_dims(coefficients, 0)

        assert coefficients.ndim == 3
        assert coefficients.shape[-1] == 2

        if range.ndim == 1:
            range = tf.expand_dims(range, 0)
        assert range.ndim == 2
        assert range.shape[-1] == 2

        assert len(range) == len(coefficients)

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
        self.points = tf.transpose(tf.linspace(range[:, 0], range[:, 1], self.granularity))
        assert self.points.shape == self.pdf.shape

        self.n = len(self.pdf)
        self.range = range
        self.coefficients = coefficients

    def _parameter_properties(self, dtype=None, num_classes=None) -> Dict[str, util.ParameterProperties]:
        return {
            "coefficients": util.ParameterProperties(),
            "range": util.ParameterProperties(),
        }

    def _sample_shape(self) -> Tuple[int, ...]:
        return ()

    def _batch_shape(self) -> Tuple[int, ...]:
        return (self.n,) if self.batched else ()

    def _sample_n(self, n: int, seed=None) -> tf.Tensor:
        # sample n points from the cdf using linear interpolation
        random_points = tf.random.uniform(
            (self.n, n),
            dtype=self.dtype,
            seed=seed
        )
        samples = tf.gather_nd(
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
        if not self.batched:
            samples = tf.squeeze(samples, -1)
        return tf.transpose(
            samples
        )

    def _prob(self, action: tf.Tensor) -> tf.Tensor:
        # interpolate between pdf values
        scaled = (action - self.range[:, 0]) / (self.range[:, 1] - self.range[:, 0]) * (self.granularity - 1)
        indices = tf.minimum(tf.cast(scaled, tf.int32), self.granularity - 2)
        delta = scaled - tf.cast(indices, self.dtype)
        return tf.gather(self.pdf, indices, axis=-1) * delta + tf.gather(self.pdf, indices + 1, axis=-1) * (1 - delta)

    def _mean(self) -> tf.Tensor:
        return tf.reduce_sum(self.points * self.pdf, axis=-1)

    def _variance(self) -> tf.Tensor:
        return tf.reduce_sum(self.points ** 2 * self.pdf, axis=-1) - self._mean() ** 2

    def _mode(self) -> tf.Tensor:
        return tf.gather_nd(
            self.points,
            tf.stack(
                (
                    tf.range(self.n),
                    tf.argmax(self.pdf, axis=-1, output_type=tf.int32)
                ),
                axis=-1
            )
        )

    def entropy(self) -> tf.Tensor:
        return -tf.reduce_sum(self.pdf * tf.math.log(self.pdf), axis=-1) / self.granularity

    def kl_divergence(self, other: FourierDistribution) -> tf.Tensor:
        pdf = tf.maximum(self.pdf, 1e-10)
        other_pdf = tf.maximum(other.pdf, 1e-10)
        return tf.reduce_sum(pdf * tf.math.log(pdf / other_pdf), axis=-1) / self.granularity
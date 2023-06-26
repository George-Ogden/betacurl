from __future__ import annotations

from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability import distributions, util
from tensorflow.keras import losses
import tensorflow as tf
import numpy as np

from typing import Dict, Optional, Tuple, Union
from dm_env.specs import BoundedArray

from ..utils import value_to_support

from .config import CombDistributionConfig
from .base import DistributionFactory

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
        self.granularity = pdf.shape[-1]
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

        self._cdf = tf.concat(
            (
                tf.cumsum(self._pdf, axis=-1, exclusive=True)[:, :-1],
                tf.maximum(1., tf.reduce_sum(self._pdf, axis=-1, keepdims=True))
            ),
            axis=-1
        )
        self._points = self.generate_coefficients(bounds=self._bounds, granularity=self.granularity)
        assert self._points.shape == self._pdf.shape

    @staticmethod
    def generate_coefficients(bounds: tf.Tensor, granularity: int) -> tf.Tensor:
        assert bounds.ndim == 2, "bounds must be a 2D tensor"
        assert bounds.shape[-1], "bounds must have 2 pairs (min and max)"
        return tf.transpose(tf.linspace(bounds[:, 0], bounds[:, 1], granularity))

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

        extra_indices = tf.stack(
            [tf.range(self.n, dtype=tf.int32)] * n,
            axis=-1
        )
        lower_indices = tf.stack(
            (
                extra_indices,
                tf.maximum(
                    0,
                    tf.cast(
                        tf.searchsorted(
                            self._cdf,
                            random_points,
                            side="right"
                        ),
                        tf.int32
                    ) - 1,
                )
            ),
            axis=-1
        )
        upper_indices = tf.stack(
            (
                extra_indices,
                tf.minimum(
                    lower_indices[..., 1] + 1,
                    self.granularity - 1
                )
            ),
            axis=-1
        )
        lower_bounds = tf.gather_nd(
            self._cdf,
            lower_indices,
        )
        upper_bounds = tf.gather_nd(
            self._cdf,
            upper_indices,
        )
        interpolation = (random_points - lower_bounds) / (upper_bounds - lower_bounds + 1e-8)
        assert tf.reduce_all(0 <= interpolation) and tf.reduce_all(interpolation <= 1), "interpolation must be between 0 and 1"
        samples = tf.transpose(
            tf.gather_nd(
                self._points,
                lower_indices
            ) * (1 - interpolation) + tf.gather_nd(
                self._points,
                upper_indices
            ) * interpolation
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

        # convert to support to get weighted coefficients
        flat_action = tf.reshape(action, -1)
        flat_pdf = tf.reshape(
            tf.tile(
                self._points[tf.newaxis],
                (len(action),) + (1,) * (action.ndim)
            ),
            (-1, self.granularity)
        )

        probability_coefficients = tf.reshape(
            value_to_support(
                flat_action,
                flat_pdf
            ),
            (-1, self.n, self.granularity,)
        )
        return tf.reshape(
            tf.reduce_sum(
                probability_coefficients * self._pdf,
                axis=-1
            ),
            action_shape
        )


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

class CombDistributionFactory(DistributionFactory):
    CONFIG_CLASS = CombDistributionConfig
    def __init__(
        self,
        move_spec: BoundedArray,
        config: CombDistributionConfig = CombDistributionConfig()
    ):
        super().__init__(move_spec, config=config)
        self.granularity = config.granularity

        self.action_support = CombDistribution.generate_coefficients(
            self.action_range.reshape(2, -1).transpose(1, 0),
            granularity=self.granularity
        )

    def _create_distribution(
        self,
        parameters: tf.Tensor,
        features: Optional[tf.Tensor] = None,
    ) -> CombDistribution:
        parameters = tf.nn.softmax(parameters, axis=-1)

        # add dirichlet noise for exploration
        dirichlet_distribution = distributions.Dirichlet(
            tf.constant(
                [self.granularity] * self.granularity,
                dtype=tf.float32
            ),
            validate_args=False
        )
        noise = dirichlet_distribution.sample(parameters.shape[:-1])
        parameters = parameters * (1 - self.noise_ratio) + noise * self.noise_ratio

        bounds = self.generate_bounds(parameters)

        return CombDistribution(
            parameters,
            bounds=bounds
        )

    @property
    def parameters_shape(self) -> Tuple[int, ...]:
        return self.action_shape + (self.granularity,)

    def parameterize(
        self,
        actions: Union[tf.Tensor, np.ndarray]
    ) -> tf.Tensor:
        """convert actions to parameters of the distribution"""
        if not isinstance(actions, tf.Tensor):
            actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        return value_to_support(
            values=actions,
            support=self.action_support
        )

    def compute_loss(
        self,
        target_parameters: tf.Tensor,
        parameters: distributions.Distribution
    ) -> tf.Tensor:
        assert isinstance(parameters, CombDistribution), "predicted distribution must be CombDistribution"
        return losses.categorical_crossentropy(
            tf.reshape(
                target_parameters,
                (-1, self.granularity),
            ),
            parameters._pdf
        )

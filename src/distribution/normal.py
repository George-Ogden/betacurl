from __future__ import annotations

from tensorflow_probability import distributions
import tensorflow as tf
import numpy as np

from dm_env.specs import BoundedArray
from typing import Tuple, Union

from .config import NormalDistributionConfig
from .base import DistributionFactory

class ClippedNormalDistribution(distributions.Normal):
    def __init__(
        self,
        loc: tf.Tensor,
        scale: tf.Tensor,
        bounds: tf.Tensor
    ):
        super().__init__(loc=loc, scale=scale)
        self.bounds = tf.cast(bounds, dtype=self.dtype)
        assert bounds.shape[-1] == 2, "the bounds must come in pairs (min-max)"
        batch_size = loc.shape
        assert self.bounds.shape[:-1] == loc.shape[1-self.bounds.ndim:], "the bounds must have the same batch size as the loc"
    
    def _sample_n(self, n, seed=None):
        sample = super()._sample_n(n, seed=seed)
        return tf.clip_by_value(
            sample,
            tf.gather(self.bounds, 0, axis=-1),
            tf.gather(self.bounds, 1, axis=-1)
        )

class NormalDistributionFactory(DistributionFactory):
    CONFIG_CLASS = NormalDistributionConfig
    def __init__(
        self,
        move_spec: BoundedArray,
        config: NormalDistributionConfig = NormalDistributionConfig()
    ):
        super().__init__(move_spec, config=config)

    def create_distribution(self, parameters: tf.Tensor) -> distributions.Normal:
        mean, std = tf.split(parameters, 2, axis=-1)
        
        std = tf.squeeze(std, axis=-1)
        std = tf.nn.softplus(std) + 1e-5

        mean = tf.squeeze(mean, axis=-1)
        mean = tf.nn.sigmoid(mean)
        mean *= self.action_range[1] - self.action_range[0]
        mean += self.action_range[0]

        action_range = np.transpose(self.action_range, (*range(1, self.action_dim), 0))
        bounds = np.tile(
            action_range,
            parameters.shape[:-self.action_dim] + (1, ) * self.action_dim
        )

        return ClippedNormalDistribution(
            loc=mean,
            scale=std,
            bounds=bounds
        )
    
    @property
    def parameters_shape(self) -> Tuple[int, ...]:
        return self.action_shape + (2,)

    def parameterize(
        self,
        actions: Union[tf.Tensor, np.ndarray]
    ) -> tf.Tensor:
        ...

    def compute_loss(
        self,
        target_parameters: tf.Tensor,
        parameters: distributions.Distribution
    ) -> tf.Tensor:
        ...
import numpy as np
from tensorflow_probability import distributions
import tensorflow as tf

from typing import Optional, Tuple, Type
from dm_env.specs import BoundedArray

from .config import DistributionConfig, SDEDistributionConfig
from .normal import NormalDistributionFactory
from .base import DistributionFactory

class SDEDistributionDecorator(DistributionFactory):
    def __init__(
        self,
        Distribution: Type[DistributionFactory],
        config: SDEDistributionConfig,
        *args, **kwargs
    ):
        self.noise_ratio = config.noise_ratio

        self.distribution = Distribution(
            config=config,
            *args, **kwargs
        )

    @property
    def parameters_shape(self) -> Tuple[int, ...]:
        return self.distribution.parameters_shape

    def create_distribution(
        self,
        parameters: tf.Tensor,
        features: Optional[tf.Tensor] = None
    ) -> distributions.Distribution:
        return self.distribution.create_distribution(parameters, features)

    def parameterize(self, actions: tf.Tensor) -> tf.Tensor:
        """convert actions to parameters of the distribution"""
        return self.distribution.parameterize(actions)

    def compute_loss(
        self,
        target_parameters: tf.Tensor,
        parameters: distributions.Distribution
    ) -> tf.Tensor:
        return self.distribution.compute_loss(target_parameters, parameters)

class NormalSDEDistributionFactory(SDEDistributionDecorator):
    CONFIG_CLASS = SDEDistributionConfig
    def __init__(
        self,
        move_spec: BoundedArray,
        config: SDEDistributionConfig = SDEDistributionConfig()
    ):
        super().__init__(NormalDistributionFactory, move_spec=move_spec, config=config)
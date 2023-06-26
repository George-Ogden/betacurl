import numpy as np
from tensorflow_probability import distributions
import tensorflow as tf

from typing import Optional, Tuple, Type
from dm_env.specs import BoundedArray

from .config import NormalSDNDistributionConfig, SDNDistributionConfig
from .normal import NormalDistributionFactory
from .base import DistributionFactory

class SDNDistributionDecorator(DistributionFactory):
    def __init__(
        self,
        Distribution: Type[DistributionFactory],
        move_spec: BoundedArray,
        config: SDNDistributionConfig,
        *args, **kwargs
    ):
        self.noise_ratio = config.noise_ratio
        self.exploration_steps = config.exploration_steps
        super().__init__(
            move_spec=move_spec,
            config=config,
        )
        config.noise_ratio = 0

        self.distribution = Distribution(
            move_spec=move_spec,
            config=config,
            *args, **kwargs
        )
        self.num_steps = 0
    
    def noise_on(self):
        super().noise_on()
        self.num_steps = 0
    
    def reset_noise(self, features: tf.Tensor):
        self.num_steps = self.exploration_steps
        self.exploration_matrix = tf.random.normal(
            shape=(np.prod(self.parameters_shape), features.shape[-1]),
            dtype=tf.float32
        ) * self.noise_ratio
    
    def generate_noise(self, features: tf.Tensor) -> tf.Tensor:
        if self.num_steps <= 0:
            self.reset_noise(features)
        self.num_steps -= 1
        features = tf.expand_dims(features, axis=-1) # expand for matmul
        noise = tf.reshape(
            self.exploration_matrix @ features,
            shape=features.shape[:-2] + self.parameters_shape # -2 after expanding
        )
        return noise

    @property
    def parameters_shape(self) -> Tuple[int, ...]:
        return self.distribution.parameters_shape

    def _create_distribution(
        self,
        parameters: tf.Tensor,
        features: Optional[tf.Tensor] = None
    ) -> distributions.Distribution:
        parameters += self.generate_noise(features)
        return self.distribution._create_distribution(parameters, features)

    def parameterize(self, actions: tf.Tensor) -> tf.Tensor:
        """convert actions to parameters of the distribution"""
        return self.distribution.parameterize(actions)

    def compute_loss(
        self,
        target_parameters: tf.Tensor,
        parameters: distributions.Distribution
    ) -> tf.Tensor:
        return self.distribution.compute_loss(target_parameters, parameters)

class NormalSDNDistributionFactory(SDNDistributionDecorator):
    CONFIG_CLASS = NormalSDNDistributionConfig
    def __init__(
        self,
        move_spec: BoundedArray,
        config: SDNDistributionConfig = SDNDistributionConfig()
    ):
        super().__init__(NormalDistributionFactory, move_spec=move_spec, config=config)
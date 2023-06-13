from __future__ import annotations

from tensorflow_probability import distributions
import tensorflow as tf
from copy import copy

from abc import ABC, abstractmethod, abstractproperty
from typing import List, Optional, Tuple
from dm_env.specs import BoundedArray

from .config import DistributionConfig

class DistributionFactory(ABC):
    CONFIG_CLASS = DistributionConfig
    def __init__(
        self,
        move_spec: BoundedArray,
        config: Optional[DistributionConfig]
    ):
        self.config = copy(config)
        self.move_spec = move_spec
    
    @abstractproperty
    def parameters_shape(self) -> Tuple[int, ...]:
        ...

    @abstractmethod
    def create_distribution(self, parameters: tf.Tensor) -> distributions.Distribution:
        ...
    
    def noise_on(self):
        """set exploration noise to 0"""
        ...
    
    def noise_off(self):
        """set exploration noise to initial value"""
        ...
    
    @abstractmethod
    def parameterize(self, actions: tf.Tensor) -> tf.Tensor:
        """convert actions to parameters of the distribution"""
        ...
    
    @abstractmethod
    def compute_loss(
        self,
        target_parameters: tf.Tensor,
        parameters: distributions.Distribution
    ) -> tf.Tensor:
        ...

    def aggregate_parameters(
        self,
        parameters: List[Tuple[tf.Tensor, int]]
    ) -> tf.Tensor:
        """return parameters for a new distribution that aggregates the given distributions weighted by their counts"""
        parameters, counts = zip(*parameters)
        parameters = tf.cast(tf.stack(parameters, axis=-1), dtype=tf.float32)
        counts = tf.constant(counts, dtype=tf.float32)
        return tf.reduce_sum(
            parameters * counts,
            axis=-1
        ) / tf.reduce_sum(counts)
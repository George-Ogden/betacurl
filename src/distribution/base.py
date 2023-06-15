from __future__ import annotations

from tensorflow_probability import distributions
import tensorflow as tf
from copy import copy
import numpy as np

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

        self.action_range = np.stack((move_spec.minimum, move_spec.maximum), axis=0, dtype=np.float32)
        self.action_shape = move_spec.shape
        self.action_dim = self.action_range.ndim
        self.noise_ratio = self.config.noise_ratio
    
    def noise_off(self):
        """set exploration noise to 0"""
        self.noise_ratio = 0

    def noise_on(self):
        """set exploration noise to initial value"""
        self.noise_ratio = self.config.noise_ratio
    
    @abstractproperty
    def parameters_shape(self) -> Tuple[int, ...]:
        ...

    @abstractmethod
    def create_distribution(
        self,
        parameters: tf.Tensor,
        features: Optional[tf.Tensor] = None
    ) -> distributions.Distribution:
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

    @staticmethod
    def aggregate_parameters(
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

    def generate_bounds(self, parameters: tf.Tensor) -> np.ndarray:
        """generate bounds for the given parameters"""
        action_range = np.transpose(self.action_range, (*range(1, self.action_dim), 0))
        bounds = np.tile(
            action_range,
            parameters.shape[:-self.action_dim] + (1, ) * self.action_dim
        )
        return bounds
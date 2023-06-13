from __future__ import annotations

from tensorflow_probability import distributions
import tensorflow as tf
from copy import copy

from abc import ABC, abstractmethod, abstractproperty
from dm_env.specs import BoundedArray
from typing import Optional, Tuple

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
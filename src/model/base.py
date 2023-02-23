from tensorflow.keras import layers
from tensorflow import keras
import numpy as np

from abc import ABCMeta, abstractclassmethod
from typing import Optional, Tuple, Union

from .config import ModelConfig

class ModelFactory(metaclass=ABCMeta):
    NAME: str = ""
    _model_count: int = 0
    CONFIG_CLASS: ModelConfig = ModelConfig
    @classmethod
    def create_model(cls, input_shape: Union[int, Tuple[int]], output_shape: Union[int, Tuple[int]], config: Optional[ModelConfig] = None) -> keras.Model:
        if config is None:
            config = cls.CONFIG_CLASS()
        return keras.Sequential(
            [
                keras.Input(np.reshape(input_shape, -1)),
                layers.Reshape((np.prod(input_shape, dtype=int),)),
                cls._create_model(np.prod(input_shape, dtype=int), np.prod(output_shape, dtype=int), config=config),
                layers.Reshape(np.reshape(output_shape, -1)),
            ]
        )

    @abstractclassmethod
    def _create_model(cls, input_size: int, output_size: int, config: Optional[ModelConfig] = ModelConfig()) -> keras.Model:
        ...

    @classmethod
    def get_name(cls):
        name = f"{cls._model_count}_{cls.NAME}"
        cls._model_count += 1
        return name

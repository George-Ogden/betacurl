from tensorflow.keras import Model

from abc import ABCMeta, abstractclassmethod
from typing import Optional

from .config import ModelConfig

class ModelFactory(metaclass=ABCMeta):
    @abstractclassmethod
    def create_model(cls, input_size: int, output_size: int, config: Optional[ModelConfig] = ModelConfig()) -> Model:
        ...
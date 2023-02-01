from .config import ModelConfig
from tensorflow.keras import Model

from abc import ABCMeta, abstractclassmethod
from typing import Optional

class ModelFactory(metaclass=ABCMeta):
    @abstractclassmethod
    def create_model(cls, input_size: int, output_size: int, config: Optional[ModelConfig] = ModelConfig()) -> Model:
        ...
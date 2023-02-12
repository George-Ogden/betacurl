from tensorflow.keras import Model

from abc import ABCMeta, abstractclassmethod
from typing import Optional

from .config import ModelConfig

class ModelFactory(metaclass=ABCMeta):
    NAME: str = ""
    _model_count: int = 0
    @abstractclassmethod
    def create_model(cls, input_size: int, output_size: int, config: Optional[ModelConfig] = ModelConfig()) -> Model:
        ...

    @classmethod
    def get_name(cls):
        name = f"{cls._model_count}_{cls.NAME}"
        cls._model_count += 1
        return name
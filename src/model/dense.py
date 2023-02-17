from tensorflow.keras import layers, Model
from tensorflow import keras

from typing import Optional

from .config import ModelConfig
from .base import ModelFactory

class DenseModelFactory(ModelFactory):
    NAME = "dense"
    CONFIG_CLASS = ModelConfig
    @classmethod
    def create_model(cls, input_size: int, output_size: int, config: Optional[ModelConfig] = ModelConfig()) -> Model:
        return keras.Sequential(name=cls.get_name(),
            layers=[
                keras.Input(shape=(input_size,)),
                layers.Dense(output_size, activation=config.output_activation)
            ]
        )

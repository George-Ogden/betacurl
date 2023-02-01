from tensorflow.keras import layers
from tensorflow import keras

from typing import Optional

from .config import ModelConfig
from .base import ModelFactory

class ConstantModel(ModelFactory):
    @classmethod
    def create_model(cls, input_size: int, output_size: int, config: Optional[ModelConfig] = ModelConfig(), constant: int = 0):
        return keras.Sequential(
            name="constant",
            layers=[
                keras.Input(shape=(input_size,)),
                layers.Dense(output_size),
                layers.Lambda(lambda x: x * 0 + constant)
            ]
        )

class ZeroModel(ConstantModel):
    @classmethod
    def create_model(cls, input_size: int, output_size: int, config: Optional[ModelConfig] = ModelConfig()):
        return super().create_model(input_size, output_size, config, constant=0)

class OneModel(ConstantModel):
    @classmethod
    def create_model(cls, input_size: int, output_size: int, config: Optional[ModelConfig] = ModelConfig()):
        return super().create_model(input_size, output_size, config, constant=1)

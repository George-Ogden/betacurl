from src.model.config import ModelConfig
from src.model.base import ModelFactory

from tensorflow.keras import layers, Model
from tensorflow import keras

from typing import Optional


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

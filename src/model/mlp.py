from tensorflow.keras import layers, Model
from tensorflow import keras

from typing import Optional

from .config import MLPModelConfig
from .base import ModelFactory

class MLPModelFactory(ModelFactory):
    NAME = "mlp"
    CONFIG_CLASS = MLPModelConfig
    @classmethod
    def create_model(cls, input_size: int, output_size: int, config: Optional[MLPModelConfig] = MLPModelConfig()) -> Model:
        return keras.Sequential(name=cls.get_name(),
            layers=[
                keras.Input(shape=(input_size,)),
                layers.Dense(config.hidden_size, activation="relu"),
                layers.Dense(output_size, activation=config.output_activation)
            ]
        )

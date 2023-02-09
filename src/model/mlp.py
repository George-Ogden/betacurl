from tensorflow.keras import layers, Model
from tensorflow import keras

from typing import Optional

from .config import MLPModelConfig
from .base import ModelFactory

class MLPModelFactory(ModelFactory):
    CONFIG_CLASS = MLPModelConfig
    @classmethod
    def create_model(cls, input_size: int, output_size: int, config: Optional[MLPModelConfig] = MLPModelConfig()) -> Model:
        return keras.Sequential(name="simple_linear",
            layers=[
                keras.Input(shape=(input_size,)),
                layers.BatchNormalization(),
                layers.Dense(config.hidden_size, activation="relu"),
                layers.Dense(output_size, activation=config.output_activation)
            ]
        )

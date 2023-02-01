from tensorflow.keras import layers, Model
from tensorflow import keras

from typing import Optional

from .config import SimpleLinearModelConfig
from .base import ModelFactory

class SimpleLinearModelFactory(ModelFactory):
    CONFIG_CLASS = SimpleLinearModelConfig
    @classmethod
    def create_model(cls, input_size: int, output_size: int, config: Optional[SimpleLinearModelConfig] = SimpleLinearModelConfig()) -> Model:
        return keras.Sequential(name="simple_linear",
            layers=[
                keras.Input(shape=(input_size,)),
                layers.BatchNormalization(),
                layers.Dense(config.hidden_size, activation="relu"),
                layers.Dense(output_size, activation=config.output_activation)
            ]
        )

from tensorflow.keras import layers, Model
from tensorflow import keras

from typing import Optional

from .config import FCNNConfig
from .base import ModelFactory

class FCNNModelFactory(ModelFactory):
    CONFIG_CLASS = FCNNConfig
    @classmethod
    def create_model(cls, input_size: int, output_size: int, config: Optional[FCNNConfig] = FCNNConfig()) -> Model:
        return keras.Sequential(name="fcnn",
            layers=[
                keras.Input(shape=(input_size,)),
                layers.BatchNormalization(),
                layers.Dense(config.hidden_size, activation="relu"),
                layers.Dropout(config.dropout),
                layers.Dense(config.hidden_size, activation="relu"),
                layers.Dropout(config.dropout),
                layers.Dense(output_size, activation=config.output_activation)
            ]
        )

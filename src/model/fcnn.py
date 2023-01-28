from src.model.config import FCNNConfig
from src.model.base import ModelFactory

from tensorflow.keras import layers, Model
from tensorflow import keras

from typing import Optional

class FCNNModelFactory(ModelFactory):
    CONFIG_CLASS = FCNNConfig
    @classmethod
    def create_model(cls, input_size: int, output_size: int, config: Optional[FCNNConfig] = FCNNConfig()) -> Model:
        return keras.Sequential(name="simple_linear",
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

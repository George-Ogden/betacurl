from tensorflow.keras import layers, Model
from tensorflow import keras

from typing import Optional

from .config import FCNNConfig
from .base import ModelFactory

class MultiLayerModelFactory(ModelFactory):
    NAME = "multi_layer_model"
    CONFIG_CLASS = FCNNConfig
    @classmethod
    def create_model(cls, input_size: int, output_size: int, config: Optional[FCNNConfig] = FCNNConfig()) -> Model:
        return keras.Sequential(name=cls.get_name(),
            layers=[
                keras.Input(shape=(input_size,)),
            ] + [
                cls.create_intermediate_layer(config) for _ in range(config.hidden_layers - 1)
            ] + [
                layers.Dense(output_size, activation=config.output_activation)
            ]
        )
    
    @classmethod
    def create_intermediate_layer(cls, config: FCNNConfig) -> keras.Sequential:
        return keras.Sequential(
            layers = [
                layers.Dense(config.hidden_size, activation="relu"),
                layers.Dropout(config.dropout)
            ]
        )

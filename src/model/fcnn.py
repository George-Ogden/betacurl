from tensorflow.keras import layers, Model
from tensorflow import keras

from typing import Optional

from .config import FCNNConfig
from .base import ModelFactory

class MultiLayerModelFactory(ModelFactory):
    CONFIG_CLASS = FCNNConfig
    @classmethod
    def create_model(cls, input_size: int, output_size: int, config: Optional[FCNNConfig] = FCNNConfig()) -> Model:
        assert config.hidden_layers >= 1
        return keras.Sequential(name="multi_layer_factory",
            layers=[
                keras.Input(shape=(input_size,)),
                layers.BatchNormalization()
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

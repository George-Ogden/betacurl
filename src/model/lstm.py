from tensorflow.keras import layers, Model
from tensorflow import keras

from typing import Optional

from .config import ModelConfig
from .base import ModelFactory

class LSTMModelFactory(ModelFactory):
    NAME = "lstm"
    CONFIG_CLASS = ModelConfig
    @classmethod
    def create_model(cls, input_size: int, output_size: int, config: Optional[ModelConfig] = ModelConfig()) -> Model:
        """config ignored but left in for consistency"""
        return keras.Sequential(name=cls.get_name(),
            layers=[
                keras.Input(shape=(None, input_size,)),
                layers.LSTM(output_size, return_sequences=True)
            ]
        )

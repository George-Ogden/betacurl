from tensorflow.keras import layers, Model
from tensorflow import keras

from typing import Optional

from .config import ModelConfig
from .base import ModelFactory

class EmbeddingFactory(ModelFactory):
    NAME = "embedding"
    CONFIG_CLASS = ModelConfig
    @classmethod
    def _create_model(cls, input_dim: int, output_dim: int, config: Optional[ModelConfig] = ModelConfig()) -> Model:
        return keras.Sequential(
            name=cls.get_name(),
            layers=[
                keras.Input(shape=()),
                layers.Embedding(input_dim, output_dim),
                layers.Activation(config.output_activation)
            ]
        )

    @classmethod
    def create_model(cls, input_shape: int, output_shape: int, config: Optional[ModelConfig] = None) -> keras.Model:
        """
        Args:
            input_shape (int): represents "vocabulary size" but called `input_shape` for consistency
            output_shape (int): represents embedding dim but called `output_shape` for consistency
        """
        if config is None:
            config = cls.CONFIG_CLASS()
        if isinstance(input_shape, tuple):
            assert len(input_shape) == 1
            input_shape, = input_shape
        if isinstance(output_shape, tuple):
            assert len(output_shape) == 1
            output_shape, = output_shape

        return cls._create_model(input_shape, output_shape, config=config)
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import numpy as np

from src.model import MultiLayerModelFactory, ModelFactory, MLPModelFactory, BEST_MODEL_FACTORY

def generic_factory_test(Factory: ModelFactory):
    model = Factory.create_model(
        input_size=2,
        output_size=3,
        config=Factory.CONFIG_CLASS(
            output_activation="sigmoid",
            hidden_size=32
        )
    )
    def find_hidden_size(layers):
        for layer in layers:
            if hasattr(layer, "units"):
                if layer.units == 32:
                    return True
            elif hasattr(layer, "layers"):
                if find_hidden_size(layer.layers):
                    return True
        return False
    assert find_hidden_size(model.layers)

    assert isinstance(model, keras.Model)
    batch = np.random.randn(20, 2)
    output = model.predict(batch)
    assert output.shape == (20, 3)
    assert (0 <= output).all() and (output <= 1).all()
    return model

def test_mlp_factory():
    generic_factory_test(MLPModelFactory)

def test_multi_layer_factory():
    generic_factory_test(MultiLayerModelFactory)

def test_config_applied_to_multi_layer_model():
    short_model = MultiLayerModelFactory.create_model(
        input_size=1,
        output_size=1,
        config=MultiLayerModelFactory.CONFIG_CLASS(
            dropout=.2,
            hidden_layers=2
        )
    )
    long_model = MultiLayerModelFactory.create_model(
        input_size=1,
        output_size=1,
        config=MultiLayerModelFactory.CONFIG_CLASS(
            dropout=.2,
            hidden_layers=3
        )
    )
    for layer in short_model.layers:
        if isinstance(layer, layers.Dropout):
            assert layer.rate == .2

    assert len(long_model.layers) > len(short_model.layers)
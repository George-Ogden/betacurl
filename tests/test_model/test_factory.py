from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import numpy as np

from src.model import DenseModelFactory, LSTMModelFactory, MultiLayerModelFactory, ModelFactory, MLPModelFactory, BEST_MODEL_FACTORY

from tests.utils import find_hidden_size

def generic_factory_test(Factory: ModelFactory):
    model = Factory.create_model(
        input_size=2,
        output_size=3,
        config=Factory.CONFIG_CLASS(
            output_activation="sigmoid",
            hidden_size=63
        )
    )
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

def test_general_distinctness():
    name1 = ModelFactory.get_name()
    name2 = ModelFactory.get_name()
    assert name1 != name2

def test_model_uniqueness():
    names = set()
    for Model in MLPModelFactory, MultiLayerModelFactory, DenseModelFactory:
        for _ in range(10):
            names.add(Model.get_name())
    assert len(names) == 30

def test_lstm_factory():
    model = LSTMModelFactory.create_model(
        input_size=2,
        output_size=3,
        config=LSTMModelFactory.CONFIG_CLASS(
            output_activation="tanh",
        )
    )

    assert isinstance(model, keras.Model)
    batch = np.random.randn(5, 10, 2)
    sequence = model.predict(batch)
    assert sequence.shape == (5, 10, 3)
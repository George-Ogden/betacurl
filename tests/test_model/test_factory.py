from tensorflow.keras import layers
from tensorflow import keras
import numpy as np

from pytest import mark

from src.model import DenseModelFactory, EmbeddingFactory, ModelConfig, ModelFactory, MLPModelFactory, MultiLayerModelFactory, BEST_MODEL_FACTORY

from tests.utils import find_hidden_size


@mark.parametrize(
    "Factory",
    [
        MLPModelFactory,
        MultiLayerModelFactory,
        DenseModelFactory,
        BEST_MODEL_FACTORY
    ]
)
def test_model_factory(Factory: ModelFactory):
    model = Factory.create_model(
        input_shape=2,
        output_shape=3,
        config=Factory.CONFIG_CLASS(
            output_activation="sigmoid",
            **(
                dict(hidden_size=63)
                if Factory != DenseModelFactory
                else dict()
            )
        )
    )
    if Factory != DenseModelFactory:
        assert find_hidden_size(model.layers)

    assert isinstance(model, keras.Model)
    batch = np.random.randn(20, 2)
    output = model.predict(batch)
    assert output.shape == (20, 3)
    assert (0 <= output).all() and (output <= 1).all()
    return model

def test_embedding_factory():
    embedding_layer = EmbeddingFactory.create_model(
        10,
        3,
        config=EmbeddingFactory.CONFIG_CLASS(
            output_activation="sigmoid"
        )
    )
    assert isinstance(embedding_layer, keras.Model)
    batch = np.arange(10, dtype=np.int32)
    output = embedding_layer(batch).numpy()
    assert output.shape == (10, 3)
    assert (0 <= output).all() and (output <= 1).all()

def test_config_applied_to_multi_layer_model():
    short_model = MultiLayerModelFactory.create_model(
        input_shape=1,
        output_shape=1,
        config=MultiLayerModelFactory.CONFIG_CLASS(
            dropout=.2,
            hidden_layers=2
        )
    )
    long_model = MultiLayerModelFactory.create_model(
        input_shape=1,
        output_shape=1,
        config=MultiLayerModelFactory.CONFIG_CLASS(
            dropout=.2,
            hidden_layers=3
        )
    )

    def check_dropout(model):
        for layer in model.layers:
            if isinstance(layer, layers.Dropout):
                assert layer.rate == .2
            elif hasattr(layer, "layers"):
                check_dropout(layer)
    check_dropout(short_model)
    check_dropout(long_model)


    def count_layers(model):
        sum = len(model.layers)
        for layer in model.layers:
            if hasattr(layer, "layers"):
                sum += count_layers(layer)
        return sum

    assert count_layers(long_model) > count_layers(short_model)

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

def test_shaped_tensors():
    model = DenseModelFactory.create_model(
        input_shape=(2, 4),
        output_shape=(3,5),
        config=ModelConfig(
            output_activation="sigmoid"
        )
    )
    batch = np.random.randn(8, 2, 4)
    output = model(batch)
    assert output.shape == (8, 3, 5)
    assert (0 <= output).numpy().all() and (output <= 1).numpy().all()

def test_empty_tensors():
    model = DenseModelFactory.create_model(
        input_shape=(),
        output_shape=(),
        config=ModelConfig(
            output_activation="sigmoid"
        )
    )
    batch = np.random.randn(8)
    output = model(batch)
    assert output.shape == (8)
    assert (0 <= output).numpy().all() and (output <= 1).numpy().all()
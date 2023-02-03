from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import numpy as np

from src.model import ModelDecorator, SimpleLinearModelFactory, SimpleLinearModelConfig, TrainingConfig, BEST_MODEL_FACTORY

from tests.config import probabilistic

config = SimpleLinearModelConfig(
    output_activation="sigmoid", hidden_size=8
)

class StubModel(ModelDecorator):
    def __init__(self):
        self.model = BEST_MODEL_FACTORY.create_model(input_size=1, output_size=1)

    def learn(self, *args, **kwargs):
        ...

def test_forward():
    model = SimpleLinearModelFactory.create_model(input_size=2, output_size=1, config=config)
    input = tf.random.normal((16, 2))
    output = model(input)
    assert output.shape == (16, 1)
    assert tf.reduce_all(output > 0)
    assert tf.reduce_all(output < 1)

def test_without_config():
    model = SimpleLinearModelFactory.create_model(input_size=2, output_size=1)
    input = tf.random.normal((16, 2))
    output = model(input)
    assert output.shape == (16, 1)

@probabilistic
def test_model_fits():
    model = StubModel()
    model.model = keras.Sequential(
        [
            keras.Input(shape=(2,)),
            layers.Dense(1)
        ]
    )

    input_data = np.random.randn(10_000, 2)
    output_data = input_data.mean(axis=-1)

    model.fit(input_data, output_data)

    test_data = np.random.randn(100, 2)
    predictions = model.model.predict(test_data).squeeze(-1)
    error = (predictions - test_data.mean(axis=-1)) ** 2
    assert error.mean() < .5

def test_override_params():
    model = StubModel()
    model.model = keras.Sequential(
        [
            keras.Input(shape=(2,)),
            layers.Dense(1)
        ]
    )

    input_data = np.random.randn(100, 2)
    output_data = input_data.mean(axis=-1)

    history = model.fit(
        input_data,
        output_data,
        training_config=TrainingConfig(
            epochs=5,
            loss="mae",
            optimizer_type="SGD"
        )
    )
    assert history.epoch == list(range(5))
    assert model.model.optimizer.name.upper() == "SGD"
    assert model.model.loss == "mae"

def test_dataset_creation():
    data = [([1., 2.], [3., 4.], 5), ([11., 12.], [13., 14.], 15)]
    dataset = StubModel.create_dataset(data)
    assert isinstance(dataset, tf.data.Dataset)
    assert len(dataset.element_spec) == 3
    for spec in dataset.element_spec:
        assert spec.dtype == tf.float32

    iterator = iter(dataset)
    first_item = [data.numpy().tolist() for data in next(iterator)]
    assert first_item == list(data[0])

    second_item = [data.numpy().tolist() for data in next(iterator)]
    assert second_item == list(data[1])

def test_model_compiles_with_args():
    model = StubModel()
    model.compile_model(
        TrainingConfig(
            lr=1e-1,
            optimizer_type="Adam",
            optimizer_kwargs=dict(
                epsilon=1e-3
            ),
            metrics=["mae", "mape"],
            loss="mae"
        )
    )

    assert model.model.optimizer.name.upper() == "ADAM"
    assert model.model.optimizer._learning_rate ==1e-1
    assert model.model.optimizer.epsilon ==1e-3
    assert model.model.loss == "mae"
    assert model.model.compiled_metrics._metrics == ["mae", "mape"]
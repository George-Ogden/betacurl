from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import numpy as np

from src.model import ModelDecorator, SimpleLinearModelFactory, SimpleLinearModelConfig, TrainingConfig

config = SimpleLinearModelConfig(
    output_activation="sigmoid", hidden_size=8
)

class StubModel(ModelDecorator):
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
    assert error.mean() < .5, error.mean()

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
    data = [([1., 2.], [3., 4.], 5.), ([11., 12.], [13., 14.], 15.)]
    dataset = StubModel.create_dataset(data)
    assert isinstance(dataset, tf.data.Dataset)
    iterator = iter(dataset)
    first_item = [data.numpy().tolist() for data in next(iterator)]
    assert first_item == list(data[0])
    second_item = [data.numpy().tolist() for data in next(iterator)]
    assert second_item == list(data[1])
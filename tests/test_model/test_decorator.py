from tensorflow.keras import layers, losses
from tensorflow import keras
import tensorflow as tf
import numpy as np

from pytest import mark

from src.model import CustomDecorator, ModelDecorator, MLPModelFactory, MLPModelConfig, TrainingConfig, BEST_MODEL_FACTORY

from tests.utils import EpochCounter

config = MLPModelConfig(
    output_activation="sigmoid", hidden_size=8
)

class StubModel(ModelDecorator):
    def __init__(self):
        self.model = BEST_MODEL_FACTORY.create_model(input_shape=1, output_shape=1)

    def learn(self, *args, **kwargs):
        ...

class CustomModel(CustomDecorator):
    def __init__(self):
        self.model = BEST_MODEL_FACTORY.create_model(input_shape=2, output_shape=())
    
    def compute_loss(self, input: tf.Tensor, output: tf.Tensor) -> tf.Tensor:
        return losses.mean_squared_error(
            self.model(
                input
            ),
            output
        )

    def learn(self, *args, **kwargs):
        ...

def test_forward():
    model = MLPModelFactory.create_model(input_shape=2, output_shape=1, config=config)
    input = tf.random.normal((16, 2))
    output = model(input)
    assert output.shape == (16, 1)
    assert tf.reduce_all(output > 0)
    assert tf.reduce_all(output < 1)

def test_without_config():
    model = MLPModelFactory.create_model(input_shape=2, output_shape=1)
    input = tf.random.normal((16, 2))
    output = model(input)
    assert output.shape == (16, 1)

@mark.flaky
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
            training_epochs=5,
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

def test_best_model_restored():
    model = StubModel()
    model.model = MLPModelFactory.create_model(1, 1, MLPModelConfig(output_activation="sigmoid", hidden_size=1))
    config = TrainingConfig(
        training_epochs=100,
        training_patience=5,
        fit_kwargs=dict(
            validation_data = (np.array((.5,)), np.array((0.,)))
        )
    )
    history = model.fit(np.array((.5,)), np.array((1.,)), training_config=config)
    val_loss = history.history["val_loss"]
    final_loss = model.model.evaluate(np.array((.5,)), np.array((0.,)))[0]
    
    assert np.argmin(val_loss) == len(val_loss) - 6
    assert np.allclose(val_loss[-6], final_loss)

@mark.flaky
def test_custom_model_fits():
    model = CustomModel()
    input_data = np.array([(x, y) for x in range(2) for y in range(2)], dtype=float)
    training_data = [((x, y), float(int(x) ^ int(y))) for x, y in input_data] * 100
    dataset = model.create_dataset(training_data)
    history = model.fit(
        dataset,
        training_config=TrainingConfig(
            lr=1e-1,
            training_epochs=20
        )
    )
    assert np.allclose(
        model.model(input_data),
        [int(x) ^ int(y) for x,y in input_data],
        atol=.1
    )

def test_custom_model_uses_config():
    model = CustomModel()
    input_data = np.array([(x, y) for x in range(2) for y in range(2)], dtype=float)
    training_data = [((x, y), float(int(x) ^ int(y))) for x, y in input_data] * 100
    dataset = model.create_dataset(training_data)
    counter = EpochCounter()
    history = model.fit(
        dataset,
        training_config=TrainingConfig(
            training_epochs=10,
            batch_size=32,
            lr=1e-1,
            training_patience=None,
            additional_callbacks=[counter],
            optimizer_type="SGD",
            optimizer_kwargs={
                "clipnorm": 1.,
                "momentum": .1,
            }
        )
    )

    optimizer = model.model.optimizer
    assert history.params["steps"] == np.ceil(len(training_data) * .9 / 32)
    assert counter.counter == 10
    assert optimizer.clipnorm == 1.
    assert optimizer.learning_rate == 1e-1
    assert optimizer.momentum == 0.1

@mark.flaky
def test_best_custom_model_restored():
    model = CustomModel()
    input_data = np.array([(x, y) for x in range(2) for y in range(2)], dtype=float)
    training_data = [((x, y), float(int(x) ^ int(y))) for x, y in input_data]
    dataset = model.create_dataset(training_data)
    counter = EpochCounter()
    history = model.fit(
        dataset,
        training_config=TrainingConfig(
            training_epochs=100,
            batch_size=32,
            lr=1e-1,
            validation_split=.25,
            training_patience=5,
            additional_callbacks=[counter],
            optimizer_type="SGD",
            optimizer_kwargs={
                "clipnorm": 1.,
                "momentum": .1,
            }
        )
    )

    val_loss = history.history["val_loss"]
    assert np.argmin(val_loss) == len(val_loss) - 6
    for item in training_data:
        input, output = item
        loss = model.compute_loss(
            np.array([input]),
            np.array([output])
        )
        if np.allclose(loss, val_loss[-6]):
            break
    else:
        assert False, "validation loss did not match expected loss"
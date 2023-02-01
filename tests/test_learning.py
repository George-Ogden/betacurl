from src.game import Arena, SamplingEvaluatingPlayer, SamplingEvaluatingPlayerConfig
from src.sampling import NNSamplingStrategy, WeightedNNSamplingStrategy
from src.evaluation import EvaluationStrategy, NNEvaluationStrategy
from src.model import ModelDecorator, TrainingConfig

from tests.utils import StubGame, BadSymetryStubGame, BadPlayer, GoodPlayer

from tensorflow.keras import layers
from tensorflow import keras
from copy import deepcopy
import numpy as np

stub_game = StubGame()
asymmetric_game = BadSymetryStubGame()
move_spec = stub_game.game_spec.move_spec
observation_spec = stub_game.game_spec.observation_spec

arena = Arena(game=stub_game, players=[GoodPlayer, BadPlayer])
result, history = arena.play_game(display=False, training=True, return_history=True)
training_data = [(*other_data, result) for *other_data, reward in history]
training_data *= 100

class StubModel(ModelDecorator):
    def learn(self, *args, **kwargs):
        ...

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

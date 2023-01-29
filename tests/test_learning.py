from src.evaluation.nn import NNEvaluationStrategy
from src.sampling.nn import NNSamplingStrategy
from src.model import BEST_MODEL_FACTORY
from src.curling import SingleEndCurlingGame
from src.io import ModelDecorator
from src.game import Arena

from tests.utils import StubGame, BadSymetryStubGame, BadPlayer, GoodPlayer

from tensorflow.keras import layers
from tensorflow import keras
import numpy as np

stub_game = StubGame()
asymmetric_game = BadSymetryStubGame()
move_spec = stub_game.game_spec.move_spec
observation_spec = stub_game.game_spec.observation_spec

arena = Arena(game=stub_game, players=[GoodPlayer, BadPlayer])
result, train_history = arena.play_game(display=False, training=True, return_history=True)
train_history = [(1 if player == 0 else -1, *other_data, result) for player, *other_data, reward in train_history]
train_history *= 1000

def test_model_fits():
    model = ModelDecorator()
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
    model = ModelDecorator()
    model.model = keras.Sequential(
        [
            keras.Input(shape=(2,)),
            layers.Dense(1)
        ]
    )

    input_data = np.random.randn(100, 2)
    output_data = input_data.mean(axis=-1)

    history = model.fit(input_data, output_data, epochs=5, loss="mae", optimizer="SGD")
    assert history.epoch == list(range(5))
    assert model.model.optimizer.name.upper() == "SGD"
    assert model.model.loss == "mae"

def test_sampler_learns():
    sampler = NNSamplingStrategy(action_spec=move_spec, observation_spec=observation_spec, latent_size=1)
    sampler.learn(train_history, stub_game.get_symmetries)
    assert (sampler.generate_actions(train_history[0][0]) > .75 * move_spec.maximum).all()

def test_evaluator_learns():
    evaluator = NNEvaluationStrategy(observation_spec=observation_spec)
    evaluator.learn(train_history, stub_game.get_symmetries)
    assert np.abs(evaluator.evaluate(train_history[0][1]) - result) < stub_game.max_move

def test_sampler_uses_augmentation():
    sampler = NNSamplingStrategy(action_spec=move_spec, observation_spec=observation_spec, latent_size=1)
    sampler.learn(train_history, asymmetric_game.get_symmetries)
    assert np.abs((sampler.generate_actions(train_history[0][0] * 0 + 1) - 1) < 1).all()
    assert np.abs((sampler.generate_actions(train_history[0][0] * 0 - 1) - 2) < 1).all()

def test_evaluator_uses_augmentation():
    evaluator = NNEvaluationStrategy(observation_spec=observation_spec)
    evaluator.learn(train_history, asymmetric_game.get_symmetries)
    assert np.abs(evaluator.evaluate(train_history[0][1] * 0 + 1) - 1) < 1
    assert np.abs(evaluator.evaluate(train_history[0][1] * 0 - 1) + 1) < 1
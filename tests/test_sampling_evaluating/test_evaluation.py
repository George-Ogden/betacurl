import numpy as np

from src.model.constant import OneModel, ZeroModel
from src.evaluation import NNEvaluationStrategy

from tests.utils import StubGame

stub_game = StubGame(6)

nn_strategy = NNEvaluationStrategy(observation_spec=stub_game.game_spec.observation_spec)
zero_strategy = NNEvaluationStrategy(observation_spec=stub_game.game_spec.observation_spec, model_factory=ZeroModel)
one_strategy = NNEvaluationStrategy(observation_spec=stub_game.game_spec.observation_spec, model_factory=OneModel)

def evaluation_strategy_test(strategy):
    observation = stub_game.get_observation()
    value = strategy.evaluate(observation)
    assert value.shape == ()
    return value

def evaluation_batch_strategy_test(strategy):
    observation = stub_game.get_observation()
    actions = strategy.evaluate(np.tile(observation, (5, 1)))
    assert actions.shape == (5,)

def test_nn_evaluation_strategy():
    evaluation_strategy_test(nn_strategy)

def test_nn_evaluation_strategy_batch():
    evaluation_batch_strategy_test(nn_strategy)

def test_nn_minimum_evaluation_strategy():
    value = evaluation_strategy_test(zero_strategy)
    assert value == 0

def test_nn_maximum_evaluation_strategy():
    value = evaluation_strategy_test(one_strategy)
    assert value == 1
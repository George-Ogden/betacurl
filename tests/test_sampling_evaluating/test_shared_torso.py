import numpy as np

from src.sampling import SharedTorsoSamplingEvaluatingStrategy

from tests.test_sampling_evaluating.test_evaluation import evaluation_strategy_test, evaluation_batch_strategy_test
from tests.test_sampling_evaluating.test_sampling import sampling_batch_strategy_test, sampling_strategy_test
from tests.utils import StubGame

stub_game = StubGame()
move_spec = stub_game.game_spec.move_spec
observation_spec = stub_game.game_spec.observation_spec

strategy = SharedTorsoSamplingEvaluatingStrategy(action_spec=move_spec, observation_spec=observation_spec)

def test_construction():
    sampler = SharedTorsoSamplingEvaluatingStrategy(action_spec=move_spec, observation_spec=observation_spec)
    outputs = sampler.model(np.expand_dims(stub_game.get_observation(), 0))
    assert len(outputs) == 2

def test_sampling():
    sampling_strategy_test(strategy)

def test_batch_sampling():
    sampling_batch_strategy_test(strategy)

def test_evaluation():
    evaluation_strategy_test(strategy)

def test_batch_evaluation():
    evaluation_batch_strategy_test(strategy)
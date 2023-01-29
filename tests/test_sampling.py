from src.sampling.range import MaxSamplingStrategy, MinSamplingStrategy
from src.sampling import RandomSamplingStrategy, NNSamplingStrategy
from src.model.constant import ZeroModel, OneModel

from tests.utils import StubGame

import tensorflow as tf
import numpy as np

stub_game = StubGame(6)

move_spec = stub_game.game_spec.move_spec
observation_spec = stub_game.game_spec.observation_spec

max_strategy = MaxSamplingStrategy(action_spec=move_spec, observation_spec=observation_spec)
min_strategy = MinSamplingStrategy(action_spec=move_spec, observation_spec=observation_spec)
random_strategy = RandomSamplingStrategy(action_spec=move_spec, observation_spec=observation_spec)
nn_strategy = NNSamplingStrategy(action_spec=move_spec, observation_spec=observation_spec)
zero_strategy = NNSamplingStrategy(action_spec=move_spec, observation_spec=observation_spec, model_factory=ZeroModel)
one_strategy = NNSamplingStrategy(action_spec=move_spec, observation_spec=observation_spec, model_factory=OneModel)

def validate_actions(actions, action_spec):
    assert (actions >= action_spec.minimum).all()
    assert (actions <= action_spec.maximum).all()

def strategy_test(strategy):
    observation = stub_game.get_observation()
    actions = strategy.generate_actions(observation)
    assert actions.shape == move_spec.shape
    validate_actions(actions, move_spec)
    stub_game.step(actions)
    return actions

def batch_strategy_test(strategy):
    observation = stub_game.get_observation()
    actions = strategy.generate_actions(observation, n=5)
    assert actions.shape == (5,) + move_spec.shape
    validate_actions(actions, move_spec)

def test_random_sampling_strategy_batch():
    strategy_test(random_strategy)

def test_random_sampling_strategy_batch():
    batch_strategy_test(random_strategy)
    
def test_nn_sampling_strategy():
    strategy_test(nn_strategy)

def test_nn_sampling_strategy_batch():
    batch_strategy_test(nn_strategy)

def test_nn_minimum_sampling_strategy():
    actions = strategy_test(zero_strategy)
    assert (actions == zero_strategy.action_range[0]).all()

def test_nn_maximum_sampling_strategy():
    one_strategy = NNSamplingStrategy(action_spec=move_spec, observation_spec=observation_spec, model_factory=OneModel)
    actions = strategy_test(one_strategy)
    assert (actions == one_strategy.action_range[1]).all()

def test_action_normalisation_is_reversible():
    actions = random_strategy.generate_actions(100)
    original_actions = actions.copy()
    processed_actions = nn_strategy.normalise_outputs(actions)
    double_processed_actions = nn_strategy.postprocess_actions(processed_actions)
    assert np.allclose(double_processed_actions, original_actions)

def test_maximum_sampling_strategy():
    actions = max_strategy.generate_actions(None)
    assert (actions == move_spec.maximum).all()

def test_batch_maximum_sampling_strategy():
    actions = max_strategy.generate_actions(None, n=22)
    assert len(actions) == 22
    assert actions.shape[1:] == move_spec.shape
    assert (actions == np.array([move_spec.maximum for _ in range(22)])).all()

def test_minimum_sampling_strategy():
    actions = min_strategy.generate_actions(None)
    assert (actions == move_spec.minimum).all()

def test_batch_minimum_sampling_strategy():
    actions = min_strategy.generate_actions(None, n=22)
    assert len(actions) == 22
    assert actions.shape[1:] == move_spec.shape
    assert (actions == np.array([move_spec.minimum for _ in range(22)])).all()
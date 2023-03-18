from dm_env import StepType
import numpy as np

from pytest import mark
import pytest

from src.player import Arena, Player, RandomPlayer
from src.game import Game, GameSpec, MujocoGame

easy_game = MujocoGame("point_mass", "easy")
hard_game = MujocoGame("cartpole", "swingup_sparse")
game = MujocoGame("pendulum", "swingup")

player = RandomPlayer(game.game_spec)

def test_game_is_game():
    assert isinstance(game, Game)

def test_game_terminates():
    assert game.reset().step_type == StepType.FIRST
    for _ in range(game.max_round - 1):
        assert game.step(player.move(game)).step_type == StepType.MID
    assert game.step(player.move(game)).step_type == StepType.LAST

def test_single_player():
    game.reset()
    assert game.to_play == 0
    for _ in range(game.max_round - 1):
        game.step(player.move(game))
        assert game.to_play == 0
        assert game.player_delta == 1

def test_discount():
    assert game.reset().discount is None
    for _ in range(game.max_round):
        assert game.step(player.move(game)).discount < 1.

def test_valid_actions_are_valid():
    game = MujocoGame("point_mass", "easy")
    player = RandomPlayer(game.game_spec)
    for i in range(1000):
        game.validate_action(player.move(game))
    pytest.raises(AssertionError, game.validate_action, np.array((.5,)))
    pytest.raises(AssertionError, game.validate_action, np.array(.5))
    pytest.raises(AssertionError, game.validate_action, np.array((0, 1., 0)))
    pytest.raises(AssertionError, game.validate_action, np.array((-2., 0.)))
    pytest.raises(AssertionError, game.validate_action, np.array((0., -2.)))
    pytest.raises(AssertionError, game.validate_action, np.array((2., 0)))
    pytest.raises(AssertionError, game.validate_action, np.array((0., 2.)))
    pytest.raises(AssertionError, game.validate_action, np.array((-2., 2.)))
    pytest.raises(AssertionError, game.validate_action, np.array((-2., -2.)))
    pytest.raises(AssertionError, game.validate_action, np.array((2., -2.)))
    pytest.raises(AssertionError, game.validate_action, np.array((2., 2.)))

def test_valid_observations_are_valid():
    observation = game.reset().observation
    assert (observation == game.get_observation()).all()
    game.validate_observation(observation)
    assert observation.dtype == game.game_spec.observation_spec.dtype

    assert game.get_observation().dtype == game.game_spec.observation_spec.dtype
    game.validate_observation(game.get_observation())

    observation = game.step(player.move(game)).observation
    assert (observation == game.get_observation()).all()
    game.validate_observation(observation)
    assert observation.dtype == game.game_spec.observation_spec.dtype

    assert game.get_observation().dtype == game.game_spec.observation_spec.dtype
    game.validate_observation(game.get_observation())

def test_clone():
    game.reset()
    for _ in range(game.max_round):
        action = player.move(game)
        expected_timestep = game.clone().step(action)
        timestep = game.step(action)
        assert timestep.step_type == expected_timestep.step_type
        assert (timestep.observation == expected_timestep.observation).all()
        assert timestep.reward == expected_timestep.reward
        assert timestep.discount == expected_timestep.discount
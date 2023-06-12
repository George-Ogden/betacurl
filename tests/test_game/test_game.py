from dm_env import StepType
from dm_env.specs import BoundedArray

import numpy as np
import pytest

from src.player import Arena, RandomPlayer
from src.game import GameSpec

from tests.utils import BadPlayer, GoodPlayer, StubGame

stub_game = StubGame(6)
random_player = RandomPlayer(stub_game.game_spec)
forced_arena = Arena([GoodPlayer, BadPlayer], stub_game)
random_arena = Arena([RandomPlayer, RandomPlayer], stub_game)


def test_stub_game_game_spec_is_correct():
    assert stub_game.game_spec == GameSpec(
        move_spec=BoundedArray(
            shape=(3,), dtype=np.float32, minimum=(0, 0, 0), maximum=(10, 10, 10)
        ),
        observation_spec=BoundedArray(
            shape=(1,), dtype=np.float32, minimum=(-30.0,), maximum=(30.0,)
        ),
    )

def test_correct_number_of_rounds_played():
    assert stub_game.reset().step_type.first()
    for i in range(5):
        assert stub_game.step(random_player.move(stub_game)).step_type.mid()
    assert stub_game.step(random_player.move(stub_game)).step_type.last()

def test_game_to_play_oscillates():
    stub_game.reset(starting_player=1)
    assert stub_game.to_play == 1
    for i in range(6):
        stub_game.step(random_player.move(stub_game))
        assert stub_game.to_play == i % 2

    stub_game.reset(starting_player=0)
    assert stub_game.to_play == 0
    for i in range(6):
        stub_game.step(random_player.move(stub_game))
        assert stub_game.to_play == 1 - (i % 2)

def test_correct_number_of_rounds_played_with_reset():
    assert stub_game.reset().step_type.first()
    for i in range(5):
        assert stub_game.step(random_player.move(stub_game)).step_type.mid()
    assert stub_game.step(random_player.move(stub_game)).step_type.last()

    assert stub_game.reset().step_type.first()
    for i in range(5):
        assert stub_game.step(random_player.move(stub_game)).step_type.mid()
    assert stub_game.step(random_player.move(stub_game)).step_type.last()

def test_clone():
    stub_game.reset(0)
    assert stub_game.to_play == 0
    for _ in range(stub_game.max_round):
        action = random_player.move(stub_game)
        expected_timestep = stub_game.clone().step(action)
        for _ in range(10):
            stub_game.clone().step(random_player.move(stub_game))
        timestep = stub_game.step(action)
        assert timestep.step_type == expected_timestep.step_type
        assert (timestep.observation == expected_timestep.observation).all()
        assert timestep.reward == expected_timestep.reward
        assert timestep.discount == expected_timestep.discount

def test_valid_actions_are_valid():
    stub_game.validate_action(random_player.move(stub_game))
    stub_game.validate_action(np.array((0, 0, 0)))
    stub_game.validate_action(np.array((10, 10, 10)))
    stub_game.validate_action(np.array((9, 4, 5)))
    pytest.raises(ValueError, stub_game.validate_action, np.array((0, 2)))
    pytest.raises(ValueError, stub_game.validate_action, np.array((3, 3, -1)))
    pytest.raises(ValueError, stub_game.validate_action, np.array((11, 3, 4)))

def test_valid_observations_are_valid():
    observation = stub_game.reset().observation
    assert (observation == stub_game.get_observation()).all()
    stub_game.validate_observation(observation)
    assert observation.dtype == stub_game.game_spec.observation_spec.dtype

    assert stub_game.get_observation().dtype == stub_game.game_spec.observation_spec.dtype
    stub_game.validate_observation(stub_game.get_observation())

    observation = stub_game.step(random_player.move(stub_game)).observation
    assert (observation == stub_game.get_observation()).all()
    stub_game.validate_observation(observation)
    assert observation.dtype == stub_game.game_spec.observation_spec.dtype

    assert stub_game.get_observation().dtype == stub_game.game_spec.observation_spec.dtype
    stub_game.validate_observation(stub_game.get_observation())

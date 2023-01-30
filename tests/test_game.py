from src.game import RandomPlayer, GameSpec, Arena

from tests.utils import StubGame, GoodPlayer, BadPlayer
from collections import Counter

from dm_env._environment import StepType
from dm_env.specs import BoundedArray

import numpy as np
import pytest

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
    assert stub_game.reset().step_type == StepType.FIRST
    for i in range(5):
        assert stub_game.step(random_player.move(stub_game)).step_type == StepType.MID
    assert stub_game.step(random_player.move(stub_game)).step_type == StepType.LAST

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
    assert stub_game.reset().step_type == StepType.FIRST
    for i in range(5):
        assert stub_game.step(random_player.move(stub_game)).step_type == StepType.MID
    assert stub_game.step(random_player.move(stub_game)).step_type == StepType.LAST

    assert stub_game.reset().step_type == StepType.FIRST
    for i in range(5):
        assert stub_game.step(random_player.move(stub_game)).step_type == StepType.MID
    assert stub_game.step(random_player.move(stub_game)).step_type == StepType.LAST

def test_sample_has_no_side_effects():
    assert stub_game.reset().step_type == StepType.FIRST
    for i in range(10):
        stub_game.sample(random_player.move(stub_game))
    for i in range(5):
        assert stub_game.step(random_player.move(stub_game)).step_type == StepType.MID
        original = list(map(float, stub_game.score))
        for i in range(10):
            stub_game.sample(random_player.move(stub_game))
            assert stub_game.score == original
    assert stub_game.step(random_player.move(stub_game)).step_type == StepType.LAST

def test_valid_actions_are_valid():
    stub_game.validate_action(random_player.move(stub_game))
    stub_game.validate_action(np.array((0, 0, 0)))
    stub_game.validate_action(np.array((10, 10, 10)))
    stub_game.validate_action(np.array((9, 4, 5)))
    pytest.raises(AssertionError, stub_game.validate_action, np.array((0, 2)))
    pytest.raises(AssertionError, stub_game.validate_action, np.array((3, 3, -1)))
    pytest.raises(AssertionError, stub_game.validate_action, np.array((11, 3, 4)))

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

def test_arena_allows_good_player_to_win():
    assert forced_arena.play_game(0) == 10 * 3

def test_good_player_always_wins():
    assert forced_arena.play_games(10) == (10, 0)

def test_arena_logs(capsys):
    forced_arena.play_game(display=True)
    expected_output = " ".join(map(repr, stub_game.score))
    captured = capsys.readouterr()
    assert captured.out.strip().endswith(expected_output)

def test_random_players_split_wins():
    wins, losses = random_arena.play_games(100)
    assert wins + losses == 100
    assert min(wins, losses) > 25

def test_predetermined_history_is_correct():
    total_reward, history = forced_arena.play_game(1, return_history=True)
    for (prev_player_id, prev_observation, prev_action, prev_reward), (next_player_id, next_observation, next_action, next_reward) in zip(history[:-1], history[1:]):
        assert (prev_player_id, next_player_id) in ((1, -1), (-1, 1))
        assert (prev_player_id == 1 and prev_reward == 10) or (prev_player_id == -1 and prev_reward == 0)
        assert (prev_player_id == 1 and (prev_action == 10).all()) or (prev_player_id == -1 and (prev_action == 0).all())
        assert prev_observation + prev_reward == next_observation
    assert np.abs(next_reward) == np.min(next_action)
    assert stub_game.score[0] - stub_game.score[1] == total_reward

def test_random_history_is_correct():
    total_reward, history = random_arena.play_game(return_history=True)
    for (prev_player_id, prev_observation, prev_action, prev_reward), (next_player_id, next_observation, next_action, next_reward) in zip(history[:-1], history[1:]):
        assert (prev_player_id, next_player_id) in ((1, -1), (-1, 1))
        assert np.abs(prev_reward) == np.min(prev_action)
        assert (prev_player_id == 1 and prev_reward >= 0) or (prev_player_id == -1 and prev_reward <= 0)
        assert (prev_observation + prev_reward - next_observation) < 1e-9
    assert np.abs(next_reward) == np.min(next_action)
    assert (stub_game.score[0] - stub_game.score[1] - total_reward) < 1e-9

def test_dummy_constructor():
    player = RandomPlayer(stub_game.game_spec)
    assert player.dummy_constructor(stub_game.game_spec) == player

def test_actions_are_preserved():
    total_reward, history = random_arena.play_game(return_history=True)
    players = Counter()
    for player_id, *data in history:
        players[player_id] += 1
    assert players.most_common(2)[0][1] == players.most_common(2)[1][1]

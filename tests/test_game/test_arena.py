from src.game import Arena, RandomPlayer

from tests.utils import StubGame, GoodPlayer, BadPlayer
from collections import Counter

import numpy as np

stub_game = StubGame(6)
random_player = RandomPlayer(stub_game.game_spec)
forced_arena = Arena([GoodPlayer, BadPlayer], stub_game)
random_arena = Arena([RandomPlayer, RandomPlayer], stub_game)

def test_arena_allows_good_player_to_win():
    assert forced_arena.play_game(0) == 10 * 3

def test_good_player_always_wins():
    assert forced_arena.play_games(10) == (10, 0)

def test_arena_logs(capsys):
    forced_arena.play_game(display=True)
    expected_output = " ".join(map(repr, stub_game.score))
    captured = capsys.readouterr()
    assert captured.out.strip().endswith(expected_output)

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

def test_actions_are_preserved():
    total_reward, history = random_arena.play_game(return_history=True)
    players = Counter()
    for player_id, *data in history:
        players[player_id] += 1
    assert players.most_common(2)[0][1] == players.most_common(2)[1][1]

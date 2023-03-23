import numpy as np

from pytest import mark

from src.player import Arena, RandomPlayer

from tests.utils import BadPlayer, GoodPlayer, StubGame

stub_game = StubGame(6)
random_player = RandomPlayer(stub_game.game_spec)
forced_arena = Arena([GoodPlayer, BadPlayer], stub_game)
random_arena = Arena([RandomPlayer, RandomPlayer], stub_game)

@mark.probabilistic
def test_random_players_split_wins():
    results = np.array(random_arena.play_games(100))
    wins = np.sum(results > 0)
    draws = np.sum(results == 0)
    losses = np.sum(results < 0)

    assert wins + draws + losses == 100
    assert min(wins, losses) + draws > 25

def test_dummy_constructor():
    player = RandomPlayer(stub_game.game_spec)
    assert player.dummy_constructor(stub_game.game_spec) == player

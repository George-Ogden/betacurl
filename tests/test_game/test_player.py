from src.game import Arena, RandomPlayer

from tests.utils import BadPlayer, GoodPlayer, StubGame

stub_game = StubGame(6)
random_player = RandomPlayer(stub_game.game_spec)
forced_arena = Arena([GoodPlayer, BadPlayer], stub_game)
random_arena = Arena([RandomPlayer, RandomPlayer], stub_game)

def test_random_players_split_wins():
    wins, losses = random_arena.play_games(100)
    assert wins + losses == 100
    assert min(wins, losses) > 25

def test_dummy_constructor():
    player = RandomPlayer(stub_game.game_spec)
    assert player.dummy_constructor(stub_game.game_spec) == player

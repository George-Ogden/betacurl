from src.player import Arena, RandomPlayer

from tests.utils import BadPlayer, GoodPlayer, StubGame

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
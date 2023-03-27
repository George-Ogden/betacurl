from src.player import MCTSPlayer, NNMCTSPlayer

from tests.utils import MDPStubGame, generic_save_load_test, generic_save_test
from tests.config import cleanup, requires_cleanup

game = MDPStubGame(6)
game.reset()

@requires_cleanup
def test_player_io_without_mcts():
    player = MCTSPlayer(
        game.game_spec
    )
    generic_save_test(player)
    generic_save_load_test(player)

@requires_cleanup
def test_player_io_with_mcts():
    player = MCTSPlayer(
        game.game_spec
    )
    player.move(game)
    generic_save_test(player)
    original, copy = generic_save_load_test(player, excluded_attrs=["mcts"])
    assert hasattr(original, "mcts")
    assert hasattr(copy, "mcts")

@requires_cleanup
def test_nn_player_io_without_model():
    player = NNMCTSPlayer(
        game.game_spec
    )
    generic_save_test(player)
    generic_save_load_test(player)
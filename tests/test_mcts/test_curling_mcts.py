from src.curling import SingleEndCurlingGame

from tests.config import probabilistic

@probabilistic
def test_random_moves_inside():
    game = SingleEndCurlingGame()
    successful = 0
    for i in range(100):
        game.reset()
        game.step(
            game.get_random_move()
        )
        successful += len(game.curling.stones)
    assert successful > 50
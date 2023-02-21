from pytest import mark

from src.curling import SingleEndCurlingGame
from src.mcts import FixedMCTS

@mark.probabilistic
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

@mark.probabilistic
def test_mcts_helps():
    game = SingleEndCurlingGame()
    mcts = FixedMCTS(game, 10)
    game.reset(0)
    for i in range(game.num_stones_per_end):
        if i % 2 == 0:
            for i in range(12):
                mcts.search(game)
            actions, probs = mcts.get_action_probs()
            game.step(actions[probs.argmax()])
        else:
            game.step(game.get_random_move())

    assert game.evaluate_position() > 0
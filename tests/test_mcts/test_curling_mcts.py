from pytest import mark

from src.mcts import FixedMCTS, WideningMCTS, WideningMCTSConfig
from src.game import SingleEndCurlingGame

@mark.flaky
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

@mark.slow
@mark.flaky
def test_fixed_mcts_helps():
    game = SingleEndCurlingGame()
    mcts = FixedMCTS(game)
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

@mark.slow
@mark.flaky
def test_widening_mcts_helps():
    game = SingleEndCurlingGame()
    mcts = WideningMCTS(
        game,
        config=WideningMCTSConfig(
            cpw=.75,
            kappa=1.
        )
    )
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
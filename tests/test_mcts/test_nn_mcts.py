from tensorflow_probability import distributions
from dm_env import TimeStep
import numpy as np

from pytest import mark
import pytest

from src.mcts import NNMCTS, NNMCTSMode, NNMCTSConfig, PolicyMCTSModel

from tests.utils import MDPStubGame, MDPSparseStubGame

class BlowUp(Exception):
    ...

class BlowUpGame(MDPSparseStubGame):
    def step(self, action: np.ndarray, display: bool = False) -> TimeStep:
        step = super().step(action, display)
        if step.step_type.last():
            raise BlowUp
        return step

blowup_game = BlowUpGame(50)
game = MDPStubGame()
model = PolicyMCTSModel(game.game_spec)

def test_end_never_reached():
    blowup_game.reset()
    mcts = NNMCTS(blowup_game)
    for i in range(100):
        mcts.search()

def test_end_reached_with_stepping():
    timestep = None
    blowup_game.reset()
    mcts = NNMCTS(blowup_game)
    while timestep is None or not timestep.step_type.last():
        for i in range(10):
            try:
                mcts.search()
            except BlowUp:
                break
        else:
            actions, probs = mcts.get_action_probs(blowup_game)
            blowup_game.step(actions[probs.argmax()])
            continue
        break
    else:
        assert False, "did not reach game end"

@mark.slow
@mark.flaky
def test_actions_use_model():
    game.reset()
    mcts = NNMCTS(game, model)
    ones = np.ones_like(game.game_spec.move_spec.minimum)
    model.generate_distribution = (lambda *args, **kwargs: distributions.Normal(ones * 2, ones)).__get__(model)
    distribution = model.generate_distribution(game.get_observation())
    actions, probs = zip(*[mcts.generate_action(game.get_observation()) for _ in range(1000)])
    assert np.allclose(distribution.mean(), np.mean(actions, axis=0), atol=.1)
    assert np.allclose(distribution.stddev(), np.std(actions, axis=0), atol=.1)

def test_config_is_used():
    blowup_game.reset()
    mcts = NNMCTS(
        blowup_game,
        config=NNMCTSConfig(
            cpuct=2.5,
            max_rollout_depth=49
        )
    )
    assert mcts.cpuct == 2.5
    mcts.search()

    mcts = NNMCTS(
        blowup_game,
        config=NNMCTSConfig(
            max_rollout_depth=50
        )
    )
    with pytest.raises(BlowUp) as e:
        mcts.search()

def test_fixed_initialisation():
    game.reset()
    mcts = NNMCTS(
        game,
        config=NNMCTSConfig(
            num_actions=2,
            cpw=1.,
            kappa=1.
        ),
        initial_mode=NNMCTSMode.FIXED
    )
    for _ in range(10):
        mcts.search(game)
    assert mcts.mode == NNMCTSMode.FIXED
    assert len(mcts.get_node(game.get_observation()).transitions) == 2

def test_wide_initialisation():
    game.reset()
    mcts = NNMCTS(
        game,
        config=NNMCTSConfig(
            num_actions=2,
            cpw=1.,
            kappa=1.
        ),
        initial_mode=NNMCTSMode.WIDENING
    )
    for _ in range(10):
        mcts.search(game)
    assert mcts.mode == NNMCTSMode.WIDENING
    assert len(mcts.get_node(game.get_observation()).transitions) >= 9

def test_fixed_to_widening_conversion():
    game.reset()
    mcts = NNMCTS(
        game,
        config=NNMCTSConfig(
            num_actions=2,
            cpw=1.,
            kappa=1.
        ),
        initial_mode=NNMCTSMode.FIXED
    )
    mcts.set_mode(NNMCTSMode.WIDENING)
    for _ in range(10):
        mcts.search(game)
    assert mcts.mode == NNMCTSMode.WIDENING
    assert len(mcts.get_node(game.get_observation()).transitions) >= 9

def test_widening_to_fixed_conversion():
    game.reset()
    mcts = NNMCTS(
        game,
        config=NNMCTSConfig(
            num_actions=2,
            cpw=1.,
            kappa=1.
        ),
        initial_mode=NNMCTSMode.WIDENING
    )
    mcts.set_mode(NNMCTSMode.FIXED)
    for _ in range(10):
        mcts.search(game)
    assert mcts.mode == NNMCTSMode.FIXED
    assert len(mcts.get_node(game.get_observation()).transitions) == 2
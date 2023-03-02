from dm_env import StepType, TimeStep
import numpy as np

from pytest import mark
import pytest

from src.mcts import MCTSModel, NNMCTS, NNMCTSConfig

from tests.utils import MDPStubGame, MDPSparseStubGame

class BlowUp(Exception):
    ...

class BlowUpGame(MDPSparseStubGame):
    def step(self, action: np.ndarray, display: bool = False) -> TimeStep:
        step = super().step(action, display)
        if step.step_type == StepType.LAST:
            raise BlowUp
        return step

blowup_game = BlowUpGame(50)
game = MDPStubGame()
model = MCTSModel(game.game_spec)

def test_end_never_reached():
    blowup_game.reset()
    mcts = NNMCTS(blowup_game)
    for i in range(100):
        mcts.search()

def test_end_reached_with_stepping():
    timestep = None
    blowup_game.reset()
    mcts = NNMCTS(blowup_game)
    while timestep is None or timestep.step_type != StepType.LAST:
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

@mark.probabilistic
def test_actions_use_model():
    game.reset()
    mcts = NNMCTS(game, model)
    distribution = model.generate_distribution(game.get_observation())
    actions, probs = zip(*[mcts.generate_action(game.get_observation()) for _ in range(1000)])
    assert np.allclose(distribution.loc, np.mean(actions, axis=0), atol=.1)
    assert np.allclose(distribution.scale, np.std(actions, axis=0), atol=.1)

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
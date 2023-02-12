from typing import Optional
import numpy as np

from src.sampling import NNSamplingStrategy, RandomSamplingStrategy, SamplingStrategy, SharedTorsoSamplingStrategy
from src.game import SamplingEvaluatingPlayer, SamplingEvaluatingPlayerConfig
from src.evaluation import EvaluationStrategy, NNEvaluationStrategy
from src.curling import SingleEndCurlingGame
from src.game import Arena, RandomPlayer

from tests.utils import StubGame, SparseStubGame
from tests.config import probabilistic, slow

class MaximumSamplingStrategy(SamplingStrategy):
    def generate_actions(self, observation: np.ndarray, n: Optional[int] = None) -> np.ndarray:
        return (super().generate_actions(observation, n) + 1) * self.action_range[1]

class InBoundsEvaluator(EvaluationStrategy):
    def evaluate(self, observations: np.ndarray) -> float:
        if observations.ndim == 1:
            return -observations[-8:].sum() * observations[0]
        else:
            return -observations[:, -8:].sum(axis=-1) * observations[:, 0]

stub_game = StubGame()
move_spec = stub_game.game_spec.move_spec
observation_spec = stub_game.game_spec.observation_spec
sparse_stub_game = SparseStubGame(2)
single_end_game = SingleEndCurlingGame()
clear_distinction = lambda game_spec: SamplingEvaluatingPlayer(
    game_spec,
    SamplingStrategyClass=RandomSamplingStrategy,
    EvaluationStrategyClass=InBoundsEvaluator,
    config=SamplingEvaluatingPlayerConfig(
        num_train_samples=10,
        num_eval_samples=100,
        epsilon=1
    )
)
player = clear_distinction(single_end_game.game_spec)

distinguishable_arena = Arena(players=[clear_distinction, RandomPlayer],game=single_end_game)

def test_construction():
    sampler = SharedTorsoSamplingStrategy(action_spec=move_spec, observation_spec=observation_spec)
    outputs = sampler.model(np.expand_dims(stub_game.get_observation(), 0))
    assert len(outputs) == 2
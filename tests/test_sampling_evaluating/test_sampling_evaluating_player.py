from typing import Optional
import numpy as np

from src.sampling import NNSamplingStrategy, RandomSamplingStrategy, SamplingStrategy
from src.game import SamplingEvaluatingPlayer, SamplingEvaluatingPlayerConfig
from src.evaluation import EvaluationStrategy, NNEvaluationStrategy
from src.curling import SingleEndCurlingGame
from src.game import Arena, RandomPlayer

from tests.utils import StubGame, SparseStubGame
from tests.config import slow

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

def test_in_bounds_evaluator():
    single_end_game.reset()
    evaluator = InBoundsEvaluator(single_end_game.game_spec.observation_spec)
    assert evaluator.evaluate(single_end_game.sample(np.array([1.41, 0, 0])).observation)\
         != evaluator.evaluate(single_end_game.sample(np.array([2, 0, 0])).observation)

def test_training_and_evaluation_matter():
    single_end_game.reset()
    player.train()
    assert player.num_samples == 10
    assert player.is_training

    player.eval()
    assert player.num_samples == 100
    assert not player.is_training

    player.train()
    assert player.num_samples == 10
    assert player.is_training

    for i in range(10):
        if (single_end_game.sample(player.move(single_end_game)).observation[-8] == 0).all():
            break
    else:
        assert False, "always in bounds while training"

    player.eval()
    single_end_game.step(player.move(single_end_game))
    assert len(single_end_game.curling.stones) > 0

def test_sampler_is_used():
    maxmimum_mover = SamplingEvaluatingPlayer(
        single_end_game.game_spec,
        SamplingStrategyClass=MaximumSamplingStrategy,
        EvaluationStrategyClass=EvaluationStrategy
    )
    move = maxmimum_mover.move(single_end_game)
    assert (move == single_end_game.game_spec.move_spec.maximum).all()

def test_evaluator_is_used():
    single_end_game.reset()
    maxmimum_mover = SamplingEvaluatingPlayer(
        single_end_game.game_spec,
        SamplingStrategyClass=MaximumSamplingStrategy,
        EvaluationStrategyClass=InBoundsEvaluator
    )
    move = maxmimum_mover.move(single_end_game)
    assert len(single_end_game.curling.stones) == 0

    random_mover = SamplingEvaluatingPlayer(
        single_end_game.game_spec,
        SamplingStrategyClass=RandomSamplingStrategy,
        EvaluationStrategyClass=InBoundsEvaluator,
        config=SamplingEvaluatingPlayerConfig(num_eval_samples=100)
    )
    random_mover.eval()
    move = random_mover.move(single_end_game)
    single_end_game.step(move)
    assert len(single_end_game.curling.stones) != 0

@slow
def test_arena_training_happens():
    results = distinguishable_arena.play_games(10, training=False)
    assert results[0] >= 8

    results = distinguishable_arena.play_games(25, training=True)
    assert min(results) >= 5

def test_eval_train_are_same_class():
    assert type(player.train()) == type(player)
    assert type(player.eval()) == type(player)

def test_picks_best_move():
    player = SamplingEvaluatingPlayer(
        stub_game.game_spec,
        SamplingStrategyClass=RandomSamplingStrategy,
        EvaluationStrategyClass=EvaluationStrategy,
    )
    arena = Arena(players=[player.dummy_constructor] * 2, game=stub_game)
    score = 0
    for i in range(100):
        arena.play_game(starting_player=i % 2, display=False)
        score += np.array(arena.game.score)
    assert (score > 100 * stub_game.max_move * stub_game.max_round / 2 * (player.num_samples - 2) / player.num_samples * (stub_game.action_size - 1) / stub_game.action_size).all()

def test_works_with_less_information():
    free_player = SamplingEvaluatingPlayer(
        sparse_stub_game.game_spec,
        SamplingStrategyClass=NNSamplingStrategy,
        EvaluationStrategyClass=NNEvaluationStrategy,
        config=SamplingEvaluatingPlayerConfig(
            num_eval_samples=10
        )
    )
    forced_player = SamplingEvaluatingPlayer(
        sparse_stub_game.game_spec,
        SamplingStrategyClass=NNSamplingStrategy,
        EvaluationStrategyClass=NNEvaluationStrategy,
        config=SamplingEvaluatingPlayerConfig(
            num_eval_samples=1
        )
    )
    arena = Arena(players=[free_player.dummy_constructor, forced_player.dummy_constructor], game=sparse_stub_game)
    wins, losses = arena.play_games(2)
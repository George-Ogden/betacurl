import tensorflow as tf

from src.evaluation import NNEvaluationStrategy
from src.sampling import GaussianSamplingStrategy, RandomSamplingStrategy, SharedTorsoSamplingEvaluatingStrategy
from src.game import Coach, SamplingEvaluatingPlayer, SharedTorsoSamplingEvaluatingPlayer
from src.curling import SingleEndCurlingGame
from src.game import Arena

from tests.test_game.test_curling import ConsistentPlayer, ConsistentLeftPlayer, ConsistentRightPlayer
from unittest.mock import patch, MagicMock

game = SingleEndCurlingGame()
good_bad_arena = Arena([ConsistentPlayer, ConsistentLeftPlayer], game)
bad_good_arena = Arena([ConsistentRightPlayer, ConsistentPlayer], game)
reward, history1 = good_bad_arena.play_game(return_history=True, display=False, starting_player=0)
reward, history2 = good_bad_arena.play_game(return_history=True, display=False, starting_player=1)
reward, history3 = bad_good_arena.play_game(return_history=True, display=False, starting_player=0)
reward, history4 = bad_good_arena.play_game(return_history=True, display=False, starting_player=1)
history = [data for history in [history1, history2, history3, history4] for data in Coach.transform_history_for_training(history[-1:])]

class StubGaussianSamplingStrategy(GaussianSamplingStrategy):
    def fit(self, dataset, config):
        self.dataset = dataset

class StubNNEvaluationStrategy(NNEvaluationStrategy):
    def fit(self, X, Y, config):
        self.x = X
        self.y = Y

def test_correct_advantage_for_se_player():
    player = SamplingEvaluatingPlayer(
        game.game_spec,
        EvaluationStrategyClass=StubNNEvaluationStrategy,
        SamplingStrategyClass=StubGaussianSamplingStrategy,
    )
    player.learn(history, lambda *x: [x])
    for observation, action, value, advantage, target_log_probs in player.sampler.dataset:
        if action[1] == 0:
            assert advantage > 0
        else:
            assert advantage < 0

@patch.object(SharedTorsoSamplingEvaluatingStrategy, "fit")
def test_correct_advantage_for_st_player(mock_fit: MagicMock):
    player = SharedTorsoSamplingEvaluatingPlayer(
        game.game_spec
    )
    player.learn(history, lambda *x: [x])
    for observation, action, value, advantage, target_log_probs in mock_fit.call_args[0][0]:
        if action[1] == 0:
            assert advantage > 0
        else:
            assert advantage < 0

@patch.object(SharedTorsoSamplingEvaluatingStrategy, "fit")
def test_correct_transformation_maintained_by_symmetries(mock_fit: MagicMock):
    player = SamplingEvaluatingPlayer(
        game.game_spec,
        EvaluationStrategyClass=StubNNEvaluationStrategy,
        SamplingStrategyClass=StubGaussianSamplingStrategy,
    )
    player.learn(history, game.get_symmetries)
    for observation, action, value, advantage, target_log_probs in player.sampler.dataset:
        if action[1] == 0:
            assert advantage > 0
        else:
            assert advantage < 0

    observations = {}
    for observation, value in zip(player.evaluator.x, player.evaluator.y):
        representation = str(tf.round(observation))
        if representation in observations:
            assert value == observations[representation]
        else:
            observations[representation] = value

    player = SharedTorsoSamplingEvaluatingPlayer(
        game.game_spec
    )
    player.learn(history, game.get_symmetries)
    for observation, action, value, advantage, target_log_probs in mock_fit.call_args[0][0]:
        if action[1] == 0:
            assert advantage > 0
        else:
            assert advantage < 0
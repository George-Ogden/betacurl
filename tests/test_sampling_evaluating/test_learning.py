from copy import deepcopy
import numpy as np

from src.game import Arena, SamplingEvaluatingPlayer, SamplingEvaluatingPlayerConfig
from src.sampling import NNSamplingStrategy, WeightedNNSamplingStrategy
from src.evaluation import EvaluationStrategy, NNEvaluationStrategy
from src.model import TrainingConfig

from tests.utils import StubGame, BadSymetryStubGame, BadPlayer, GoodPlayer
from tests.config import slow

stub_game = StubGame()
asymmetric_game = BadSymetryStubGame()
move_spec = stub_game.game_spec.move_spec
observation_spec = stub_game.game_spec.observation_spec

arena = Arena(game=stub_game, players=[GoodPlayer, BadPlayer])
result, history = arena.play_game(display=False, training=True, return_history=True)
training_data = [(*other_data, result) for *other_data, reward in history]
training_data *= 100

def test_sampler_learns():
    sampler = NNSamplingStrategy(action_spec=move_spec, observation_spec=observation_spec, latent_size=1)
    sampler.learn(training_data, stub_game.get_symmetries)
    assert (sampler.generate_actions(training_data[0][0]) > .75 * move_spec.maximum).all()

def test_evaluator_learns():
    evaluator = NNEvaluationStrategy(observation_spec=observation_spec)
    evaluator.learn(training_data, stub_game.get_symmetries)
    assert np.abs(evaluator.evaluate(training_data[0][1]) - result) < stub_game.max_move

def test_sampler_uses_augmentation():
    sampler = NNSamplingStrategy(action_spec=move_spec, observation_spec=observation_spec, latent_size=1)
    sampler.learn(training_data, asymmetric_game.get_symmetries)
    assert np.abs((sampler.generate_actions(training_data[0][0] * 0 + 1) - 1) < 1).all()
    assert np.abs((sampler.generate_actions(training_data[0][0] * 0 - 1) - 2) < 1).all()

def test_evaluator_uses_augmentation():
    evaluator = NNEvaluationStrategy(observation_spec=observation_spec)
    evaluator.learn(training_data, asymmetric_game.get_symmetries)
    assert np.abs(evaluator.evaluate(training_data[0][1] * 0 + 1) - 1) < 1
    assert np.abs(evaluator.evaluate(training_data[0][1] * 0 - 1) + 1) < 1

@slow
def test_weighted_sampling_improves_on_normal_sampling():
    total_wins = 0
    for _ in range(5):
        weighted_player = SamplingEvaluatingPlayer(
            game_spec=stub_game.game_spec, 
            SamplingStrategyClass=WeightedNNSamplingStrategy,
            EvaluationStrategyClass=EvaluationStrategy,
            config=SamplingEvaluatingPlayerConfig(
                num_eval_samples=1,
                num_train_samples=1
            )
        )
        regular_player = SamplingEvaluatingPlayer(
            game_spec=stub_game.game_spec, 
            SamplingStrategyClass=NNSamplingStrategy,
            EvaluationStrategyClass=EvaluationStrategy,
            config=SamplingEvaluatingPlayerConfig(
                num_eval_samples=1,
                num_train_samples=1
            )
        )

        stub_game.reset()
        noisy_observation = regular_player.sampler.add_noise_to_observations(np.expand_dims(stub_game.get_observation(), 0))
        weighted_player.sampler.model = deepcopy(regular_player.sampler.model)
        assert (weighted_player.sampler.model(noisy_observation) == regular_player.sampler.model(noisy_observation)).numpy().all()
        
        training_config = TrainingConfig(epochs=2)
        weighted_player.learn(training_data[:100], augmentation_function=stub_game.get_symmetries, training_config=training_config)
        assert not (weighted_player.sampler.model(noisy_observation) == regular_player.sampler.model(noisy_observation)).numpy().all()

        regular_player.learn(training_data[:100], augmentation_function=stub_game.get_symmetries, training_config=training_config)

        arena = Arena(players=[weighted_player.dummy_constructor, regular_player.dummy_constructor], game=stub_game)
        wins, losses = arena.play_games(10)
        assert wins >= 5
        total_wins += wins
    assert total_wins > 25
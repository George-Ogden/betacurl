from src.game import SamplingEvaluatingPlayer, SamplingEvaluatingPlayerConfig, RandomPlayer
from src.evaluation import NNEvaluationStrategy, EvaluationStrategy
from src.sampling import NNSamplingStrategy, RandomSamplingStrategy
from src.curling import SingleEndCurlingGame
from src.io import SaveableObject, SaveableModel

from copy import copy, deepcopy
import os

from tests.utils import SAVE_DIR, requires_cleanup, cleanup

game = SingleEndCurlingGame()
observation_spec = game.game_spec.observation_spec
move_spec = game.game_spec.move_spec

def generic_save_test(object: SaveableObject):
    object.save(SAVE_DIR)

    assert os.path.exists(SAVE_DIR)
    assert os.path.exists(os.path.join(SAVE_DIR, object.DEFAULT_FILENAME))
    assert os.path.getsize(os.path.join(SAVE_DIR, object.DEFAULT_FILENAME)) > 0

def generic_model_test(model: SaveableModel):
    generic_save_test(model)

    assert os.path.exists(os.path.join(SAVE_DIR, model.DEFAULT_MODEL_FILE))
    assert os.path.getsize(os.path.join(SAVE_DIR, model.DEFAULT_MODEL_FILE)) > 0

def save_load(object: SaveableObject) -> SaveableObject:
    object.save(SAVE_DIR)
    new_object = type(object).load(SAVE_DIR)
    assert type(new_object) == type(object)
    return new_object

@requires_cleanup
def test_player_saves():
    random_player = RandomPlayer(game.game_spec)
    generic_save_test(random_player)

@requires_cleanup
def test_player_saves_no_side_effects():
    random_player = RandomPlayer(game.game_spec)
    minimum = random_player.minimum.copy()
    maximum = random_player.maximum.copy()
    generic_save_test(random_player)
    assert (random_player.minimum == minimum).all()
    assert (random_player.maximum == maximum).all()

@requires_cleanup
def test_player_loads_same():
    player1 = RandomPlayer(game.game_spec)
    player2 = save_load(player1)
    assert (player1.minimum == player2.minimum).all()
    assert (player1.maximum == player2.maximum).all()

@requires_cleanup
def test_sampler_saves():
    sampler = RandomSamplingStrategy(action_spec=move_spec, observation_spec=observation_spec)
    generic_save_test(sampler)

@requires_cleanup
def test_sampler_loads_same():
    old_sampler = RandomSamplingStrategy(action_spec=move_spec, observation_spec=observation_spec)
    new_sampler = save_load(old_sampler)
    assert (old_sampler.action_range == new_sampler.action_range).all()
    assert (old_sampler.observation_range == new_sampler.observation_range).all()

@requires_cleanup
def test_evaluator_saves():
    evaluator = EvaluationStrategy(observation_spec=observation_spec)
    generic_save_test(evaluator)

@requires_cleanup
def test_evaluator_loads_same():
    old_evaluator = EvaluationStrategy(observation_spec=observation_spec)
    new_evaluator = save_load(old_evaluator)
    assert (old_evaluator.observation_range == new_evaluator.observation_range).all()

@requires_cleanup
def test_nn_sampler_saves():
    sampler = NNSamplingStrategy(action_spec=move_spec, observation_spec=observation_spec)
    generic_model_test(sampler)

@requires_cleanup
def test_nn_sampler_saves_no_side_effects():
    sampler = NNSamplingStrategy(action_spec=move_spec, observation_spec=observation_spec)
    config = deepcopy(sampler.model.get_config())
    generic_model_test(sampler)
    assert config == sampler.model.get_config()

@requires_cleanup
def test_nn_sampler_loads_same():
    old_sampler = NNSamplingStrategy(action_spec=move_spec, observation_spec=observation_spec, latent_size=10)
    new_sampler: NNSamplingStrategy = save_load(old_sampler)
    assert new_sampler.latent_size == 10
    assert new_sampler.model.get_config() == old_sampler.model.get_config()

@requires_cleanup
def test_nn_evaluator_saves():
    evaluator = NNEvaluationStrategy(observation_spec=observation_spec)
    generic_model_test(evaluator)

@requires_cleanup
def test_nn_evaluator_saves_no_side_effects():
    evaluator = NNEvaluationStrategy(observation_spec=observation_spec)
    config = deepcopy(evaluator.model.get_config())
    generic_model_test(evaluator)
    assert config == evaluator.model.get_config()

@requires_cleanup
def test_nn_evaluator_loads_same():
    old_evaluator = NNEvaluationStrategy(observation_spec=observation_spec)
    new_evaluator: NNEvaluationStrategy = save_load(old_evaluator)
    assert (new_evaluator.observation_range == old_evaluator.observation_range).all()
    assert new_evaluator.model.get_config() == old_evaluator.model.get_config()

def test_evaluator_and_sampler_save_different_files():
    assert NNEvaluationStrategy.DEFAULT_MODEL_FILE != NNSamplingStrategy.DEFAULT_MODEL_FILE

@requires_cleanup
def test_player_saves():
    player = SamplingEvaluatingPlayer(
        game.game_spec,
        SamplingStrategyClass=NNSamplingStrategy,
        EvaluationStrategyClass=NNEvaluationStrategy,
    )
    generic_save_test(player)

@requires_cleanup
def test_player_save_no_side_effects():
    player = SamplingEvaluatingPlayer(
        game.game_spec,
        SamplingStrategyClass=NNSamplingStrategy,
        EvaluationStrategyClass=NNEvaluationStrategy,
    )
    evaluator_config = deepcopy(player.evaluator.model.get_config())
    sampler_config = deepcopy(player.sampler.model.get_config())

    generic_save_test(player)

    assert player.evaluator.model.get_config() == evaluator_config
    assert player.sampler.model.get_config() == sampler_config

@requires_cleanup
def test_state_is_restored():
    player1 = SamplingEvaluatingPlayer(
        game.game_spec,
        SamplingStrategyClass=NNSamplingStrategy,
        EvaluationStrategyClass=NNEvaluationStrategy,
        config=SamplingEvaluatingPlayerConfig(
            num_train_samples=2,
            num_eval_samples=3,
            epsilon=.5
        )
    )
    player2 = save_load(player1)

    assert player1.config == player2.config
    assert type(player1.evaluator) == type(player2.evaluator)
    assert type(player1.sampler) == type(player2.sampler)
    assert player1.evaluator.model.get_config() == player2.evaluator.model.get_config()
    assert player1.sampler.model.get_config() == player2.sampler.model.get_config()
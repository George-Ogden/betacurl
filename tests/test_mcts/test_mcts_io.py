import tensorflow as tf
import numpy as np

from src.game import Arena, MCTSPlayer, NNMCTSPlayer
from src.mcts import MCTSModel, SamplingMCTSModel

from tests.utils import MDPStubGame, generic_save_load_test, generic_save_test, save_load
from tests.config import cleanup, requires_cleanup

game = MDPStubGame(6)
arena = Arena([MCTSPlayer] * 2, game)
result, history = arena.play_game(return_history=True)
game.reset()

@requires_cleanup
def test_player_io_without_mcts():
    player = MCTSPlayer(
        game.game_spec
    )
    generic_save_test(player)
    generic_save_load_test(player)

@requires_cleanup
def test_player_io_with_mcts():
    player = MCTSPlayer(
        game.game_spec
    )
    player.move(game)
    generic_save_test(player)
    original, copy = generic_save_load_test(player, excluded_attrs=["mcts"])
    assert hasattr(original, "mcts")
    assert hasattr(copy, "mcts")

@requires_cleanup
def test_nn_player_io_without_model():
    player = NNMCTSPlayer(
        game.game_spec
    )
    generic_save_test(player)
    generic_save_load_test(player)

@requires_cleanup
def test_mcts_model_io():
    game.reset()
    model = MCTSModel(game.game_spec)
    model.predict_values(game.get_observation())
    model.generate_distribution(game.get_observation())

    generic_save_test(model)
    copy = save_load(model)

    observation = np.random.randn(*game.game_spec.observation_spec.shape)
    assert tf.reduce_all(model.generate_distribution(observation).loc == copy.generate_distribution(observation).loc)
    assert tf.reduce_all(model.generate_distribution(observation).scale == copy.generate_distribution(observation).scale)
    assert tf.reduce_all(model.predict_values(observation) == copy.predict_values(observation))

@requires_cleanup
def test_mcts_sampling_model_io():
    game.reset()
    model = SamplingMCTSModel(game.game_spec)
    model.predict_values(game.get_observation())
    model.generate_distribution(game.get_observation())

    generic_save_test(model)
    copy = save_load(model)

    observation = np.random.randn(*game.game_spec.observation_spec.shape)
    assert tf.reduce_all(model.generate_distribution(observation).loc == copy.generate_distribution(observation).loc)
    assert tf.reduce_all(model.generate_distribution(observation).scale == copy.generate_distribution(observation).scale)
    assert tf.reduce_all(model.predict_values(observation) == copy.predict_values(observation))

@requires_cleanup
def test_nn_player_io_with_model():
    player = NNMCTSPlayer(
        game.game_spec
    )
    player.move(game)
    player.learn(history, game.no_symmetries)
    player.move(game)

    model = player.model
    assert model.model is not None
    model.predict_values(game.get_observation())
    model.generate_distribution(game.get_observation())

    generic_save_test(player)
    
    copy = save_load(player).model
    
    observation = np.random.randn(*game.game_spec.observation_spec.shape)
    assert tf.reduce_all(model.generate_distribution(observation).loc == copy.generate_distribution(observation).loc)
    assert tf.reduce_all(model.generate_distribution(observation).scale == copy.generate_distribution(observation).scale)
    assert tf.reduce_all(model.predict_values(observation) == copy.predict_values(observation))
    assert copy.model is not None
import tensorflow as tf
import numpy as np

from src.mcts import DiffusionMCTSModel, ReinforceMCTSModel
from src.player import NNMCTSPlayer

from tests.utils import MDPStubGame, generic_save_test, save_load
from tests.config import cleanup, requires_cleanup

game = MDPStubGame(6)
game.reset()

@requires_cleanup
def test_mcts_model_io():
    game.reset()
    model = ReinforceMCTSModel(game.game_spec)
    model.predict_values(game.get_observation())
    model.generate_distribution(game.get_observation())

    generic_save_test(model)
    copy = save_load(model)

    observation = np.random.randn(*game.game_spec.observation_spec.shape)
    assert np.allclose(model.generate_distribution(observation).kl_divergence(copy.generate_distribution(observation)).numpy(), 0.)
    assert tf.reduce_all(model.predict_values(observation) == copy.predict_values(observation))

@requires_cleanup
def test_mcts_diffusion_model_io():
    game.reset()
    model = DiffusionMCTSModel(game.game_spec)
    model.predict_values(game.get_observation())
    model.generate_distribution(game.get_observation())

    generic_save_test(model)
    copy = save_load(model)

    observation = np.random.randn(1, *game.game_spec.observation_spec.shape)
    noise = np.random.randn(1, *game.game_spec.move_spec.shape)
    timestep = np.zeros((1,), dtype=int)
    assert np.allclose(model.diffusion_model([noise, observation, timestep]), copy.diffusion_model([noise, observation, timestep]))
    assert tf.reduce_all(model.predict_values(observation) == copy.predict_values(observation))

@requires_cleanup
def test_nn_player_io_with_model():
    player = NNMCTSPlayer(
        game.game_spec
    )
    player.model = player.create_model()
    player.move(game)

    model = player.model
    assert model.model is not None
    model.predict_values(game.get_observation())
    model.generate_distribution(game.get_observation())

    generic_save_test(player)

    copy = save_load(player).model

    observation = np.random.randn(*game.game_spec.observation_spec.shape)
    assert np.allclose(model.generate_distribution(observation).kl_divergence(copy.generate_distribution(observation)).numpy(), 0.)
    assert tf.reduce_all(model.predict_values(observation) == copy.predict_values(observation))
    assert copy.model is not None
import tensorflow as tf
import numpy as np

from src.mcts import DiffusionMCTSModel, FourierMCTSModel, ReinforceMCTSModel
from src.player import NNMCTSPlayer

from tests.utils import MDPStubGame, generic_save_test, save_load
from tests.config import cleanup, requires_cleanup

training_data = [((-1)**i, np.array((.5 * ((i + 1) // 2),)), np.array((.0,) if i % 2 else (.5,)), 1.5, [(np.array((.0,) if i % 2 else (.5,)), (-1.)**i)]) for i in range(6)]
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
def test_mcts_fourier_model_io():
    action_size = MDPStubGame.action_size
    MDPStubGame.action_size = 1
    game = MDPStubGame(6)
    MDPStubGame.action_size = action_size
    game.reset()
    
    model = FourierMCTSModel(game.game_spec)
    model.predict_values(game.get_observation())
    model.generate_distribution(game.get_observation())

    generic_save_test(model)
    copy = save_load(model)

    observation = np.random.randn(*game.game_spec.observation_spec.shape)
    assert np.allclose(model.generate_distribution(observation).kl_divergence(copy.generate_distribution(observation)).numpy(), 0.)
    assert tf.reduce_all(model.predict_values(observation) == copy.predict_values(observation))

@requires_cleanup
def test_nn_player_io_with_model():
    player = NNMCTSPlayer(
        game.game_spec
    )
    player.move(game)
    player.learn(training_data * 10, game.no_symmetries)
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
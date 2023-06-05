import tensorflow as tf
import numpy as np

from typing import Type
from pytest import mark

from src.mcts import MCTSModel, PolicyMCTSModel, PPOMCTSModel
from src.player import NNMCTSPlayer

from tests.utils import MDPStubGame, generic_save_test, save_load
from tests.config import cleanup, requires_cleanup

game = MDPStubGame(6)
game.reset()

@requires_cleanup
@mark.parametrize("Model", [PolicyMCTSModel, PPOMCTSModel])
def test_mcts_model_io(Model: Type[MCTSModel]):
    game.reset()
    model = Model(game.game_spec)
    model.noise_ratio = 0.
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
    player.model = player.create_model()
    player.move(game)

    model = player.model
    model.noise_ratio = 0.
    assert model.model is not None
    model.predict_values(game.get_observation())
    model.generate_distribution(game.get_observation())

    generic_save_test(player)

    copy = save_load(player).model

    observation = np.random.randn(*game.game_spec.observation_spec.shape)
    assert np.allclose(model.generate_distribution(observation).kl_divergence(copy.generate_distribution(observation)).numpy(), 0.)
    assert tf.reduce_all(model.predict_values(observation) == copy.predict_values(observation))
    assert copy.model is not None
import tensorflow as tf
import numpy as np

from pytest import mark

from src.mcts import MCTSModel, MCTSModelConfig
from src.model import TrainingConfig
from src.game import Arena

from tests.utils import MDPStubGame, BadPlayer, GoodPlayer

max_move = MDPStubGame.max_move
MDPStubGame.max_move = .5
stub_game = MDPStubGame()
stub_game.max_move = MDPStubGame.max_move
MDPStubGame.max_move = max_move
game_spec = stub_game.game_spec

arena = Arena(game=stub_game, players=[GoodPlayer, BadPlayer])
result, history = arena.play_game(display=False, training=True, return_history=True)
training_data = [(*other_data, result) for *other_data, reward in history]
training_data *= 100

def test_correct_advantages():
    model = MCTSModel(
        game_spec,
        config=MCTSModelConfig(
            vf_coeff=0.,
            ent_coeff=0.
        )
    )

    for (player, observation, action, reward) in training_data[:len(training_data)//100]:
        for (augmented_player, augmented_observation, augmented_action, augmented_reward) in (stub_game.get_symmetries(player, observation, action, reward)):
            advantage = model.compute_advantages(
                players=np.array([augmented_player], dtype=np.float32),
                observations=np.array([augmented_observation], dtype=np.float32),
                rewards=np.array([augmented_reward], dtype=np.float32)
            )
            if (augmented_action > 0).all():
                assert advantage > 0
            else:
                assert advantage < 0

@mark.probabilistic
def test_policy_learns():
    model = MCTSModel(
        game_spec,
        config=MCTSModelConfig(
            vf_coeff=0.,
            ent_coeff=0.,
        )
    )

    model.learn(training_data, stub_game.get_symmetries)

    assert tf.reduce_all(model.generate_distribution(training_data[0][1]).loc > .75 * game_spec.move_spec.maximum)

@mark.probabilistic
def test_value_learns():
    model = MCTSModel(
        game_spec,
        config=MCTSModelConfig(
            vf_coeff=1000.,
            ent_coeff=0.,
        )
    )

    model.learn(training_data, stub_game.get_symmetries)

    assert np.abs(model.predict_values(training_data[0][1]) - result) < stub_game.max_move

@mark.probabilistic
def test_entropy_increases():
    model = MCTSModel(
        game_spec,
        config=MCTSModelConfig(
            vf_coeff=0.,
            ent_coeff=1000.,
        )
    )

    model.learn(training_data, stub_game.get_symmetries)
    assert tf.reduce_all(model.generate_distribution(training_data[0][1]).scale > 1.)

@mark.probabilistic
def test_model_losses_converge():
    model = MCTSModel(
        game_spec,
        config=MCTSModelConfig(
            vf_coeff=1.,
            ent_coeff=.1,
        )
    )

    model.learn(
        training_data,
        stub_game.get_symmetries,
        training_config=TrainingConfig(
            training_epochs=100
        )
    )

    distribution = model.generate_distribution(training_data[0][1])
    assert np.abs(model.predict_values(training_data[0][1]) - result) < stub_game.max_move
    assert tf.reduce_any(distribution.scale > .1)
    assert tf.reduce_all(distribution.loc > .75 * game_spec.move_spec.maximum)
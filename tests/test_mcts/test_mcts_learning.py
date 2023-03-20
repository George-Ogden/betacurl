import tensorflow as tf
import numpy as np

from pytest import mark

from src.mcts import MCTSModel, MCTSModelConfig
from src.model import TrainingConfig

from tests.utils import MDPStubGame, StubGame

max_move = MDPStubGame.max_move
MDPStubGame.max_move = .5
stub_game = MDPStubGame(6)
stub_game.max_move = MDPStubGame.max_move
MDPStubGame.max_move = max_move
game_spec = stub_game.game_spec

result = 1.5
training_data = [((-1)**i, np.array((.5 * ((i + 1) // 2),)), np.array((.0,) if i % 2 else (.5,)), result, [(np.array((.0,) if i % 2 else (.5,)), (-1.)**i)]) for i in range(6)]
training_data *= 100

@mark.probabilistic
def test_policy_learns():
    model = MCTSModel(
        game_spec,
        config=MCTSModelConfig(
            vf_coeff=0.,
            ent_coeff=0.,
        ),
    )
    model.learn(training_data, stub_game.get_symmetries)

    assert tf.reduce_all(model.generate_distribution(training_data[0][1]).loc > .5 * game_spec.move_spec.maximum)

@mark.probabilistic
def test_value_learns():
    model = MCTSModel(
        game_spec,
        config=MCTSModelConfig(
            vf_coeff=1000,
            ent_coeff=0.,
        ),
    )

    model.learn(training_data, stub_game.no_symmetries)

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

    model.learn(training_data, stub_game.no_symmetries)
    assert tf.reduce_all(model.generate_distribution(training_data[0][1]).scale > 1.)

@mark.probabilistic
def test_model_losses_converge():
    model = MCTSModel(
        game_spec,
        config=MCTSModelConfig(
            vf_coeff=100.,
            ent_coeff=1e-4,
        )
    )

    model.learn(
        training_data,
        stub_game.no_symmetries,
        training_config=TrainingConfig(
            training_epochs=10,
            lr=1e-3
        )
    )

    distribution = model.generate_distribution(training_data[0][1])
    assert np.abs(model.predict_values(training_data[0][1]) - result) < stub_game.max_move
    assert tf.reduce_any(distribution.scale > .1)
    assert tf.reduce_all(distribution.loc > .75 * game_spec.move_spec.maximum)

@mark.slow
@mark.probabilistic
def test_model_learns_from_multiple_actions():
    game = StubGame(2)
    game.reset(0)
    move = np.ones(game_spec.move_spec.shape) / 10
    training_data = [(
        1, game.get_observation(), 7 * move, 1., [
            (move * 3, -1.),
            (move * 5, 0.),
            (move * 7, 1.)
        ],
    )]
    game.step(move * 7)
    training_data.append((
        0, game.get_observation(), 6 * move, 1., [
            (move * 4, -1.),
            (move * 6, 1.)
        ],
    ))
    training_data *= 100

    model = MCTSModel(game_spec)
    model.learn(
        training_data, game.get_symmetries
    )

    assert tf.reduce_all(model.generate_distribution(game.get_observation()).loc > .5)
    game.reset(0)
    assert tf.reduce_all(model.generate_distribution(game.get_observation()).loc > .5)
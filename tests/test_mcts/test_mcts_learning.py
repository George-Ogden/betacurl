import tensorflow as tf
import numpy as np

from pytest import mark
import wandb

from src.mcts import MCTSModel, MCTSModelConfig, PPOMCTSModel, PPOMCTSModelConfig
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

def test_ppo_model_stats(monkeypatch):
    logs = []
    def log(data, *args, **kwargs):
        if "train" in data or "val" in data:
            logs.append(data)

    monkeypatch.setattr(wandb, "log", log)
    model = PPOMCTSModel(
        game_spec
    )
    model.learn(training_data, stub_game.get_symmetries)

    expected_keys = ["loss", "value_loss", "policy_loss", "entropy_loss", "entropy"]

    assert len(logs) > 0
    for key in expected_keys:
        for data in logs:
            assert key in data["train"]
            assert key in data["val"]

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
            vf_coeff=.5,
            ent_coeff=0.,
        )
    )

    model.learn(
        training_data,
        stub_game.no_symmetries,
        training_config=TrainingConfig(
            training_epochs=20,
            lr=1e-2
        )
    )

    distribution = model.generate_distribution(training_data[0][1])
    assert np.abs(model.predict_values(training_data[0][1]) - result) < stub_game.max_move
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

@mark.probabilistic
def test_ppo_model_losses_converge():
    model = PPOMCTSModel(
        game_spec,
        config=PPOMCTSModelConfig(
            vf_coeff=.5,
            ent_coeff=0.,
            target_kl=None
        )
    )

    for _ in range(10):
        model.learn(
            training_data,
            stub_game.no_symmetries,
            training_config=TrainingConfig(
                training_epochs=10,
                lr=1e-2
            )
        )

    distribution = model.generate_distribution(training_data[0][1])
    assert np.abs(model.predict_values(training_data[0][1]) - result) < stub_game.max_move
    assert tf.reduce_all(distribution.loc > .75 * game_spec.move_spec.maximum)

@mark.probabilistic
def test_kl_divergence_without_early_stopping():
    model = PPOMCTSModel(
        game_spec=game_spec,
        config=PPOMCTSModelConfig(
            target_kl=None,
            vf_coeff=0.,
            ent_coeff=0.
        )
    )
    training_sample = training_data[0]
    initial_distribution = model.generate_distribution(
        training_sample[1]
    )
    training_data[-1] = [(initial_distribution.mean() + 100, 1.), (initial_distribution.mean() - 100, -1.)]
    model.learn(
        training_data=[training_sample] * 100,
        augmentation_function=stub_game.no_symmetries,
        training_config=TrainingConfig(
            training_epochs=100,
            lr=1e-1,
            batch_size=100
        )
    )
    assert tf.reduce_mean(initial_distribution.kl_divergence(
        model.generate_distribution(training_sample[1])
    )) > 5
    

@mark.probabilistic
def test_kl_divergence_with_early_stopping():
    model = PPOMCTSModel(
        game_spec=game_spec,
        config=PPOMCTSModelConfig(
            target_kl=2.,
            vf_coeff=0.,
            ent_coeff=0.
        )
    )
    training_sample = training_data[0]
    initial_distribution = model.generate_distribution(
        training_sample[1]
    )
    training_data[-1] = [(initial_distribution.mean() + 100, 1.), (initial_distribution.mean() - 100, -1.)]
    model.learn(
        training_data=[training_sample] * 10,
        augmentation_function=stub_game.no_symmetries,
        training_config=TrainingConfig(
            training_epochs=1000,
            lr=1e-3,
            batch_size=1
        )
    )
    assert tf.reduce_mean(initial_distribution.kl_divergence(
        model.generate_distribution(training_sample[1])
    )) < 5
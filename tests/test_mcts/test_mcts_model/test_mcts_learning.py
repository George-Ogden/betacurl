import tensorflow as tf
import numpy as np

from pytest import mark
import wandb

from src.mcts import PolicyMCTSModel, PolicyMCTSModelConfig, PPOMCTSModel, PPOMCTSModelConfig, ReinforceMCTSModel, ReinforceMCTSModelConfig
from src.model import MLPModelFactory, MLPModelConfig, TrainingConfig

from tests.utils import MDPStubGame, StubGame

max_move = MDPStubGame.max_move
MDPStubGame.max_move = 1.5
stub_game = MDPStubGame(6)
stub_game.max_move = MDPStubGame.max_move
MDPStubGame.max_move = max_move
game_spec = stub_game.game_spec
move = np.ones(game_spec.move_spec.shape)

result = 3.
training_data = [((-1)**i, np.array((1.25 * ((i + 1) // 2),)), (.25 * move) if i % 2 else (1.25 * move), result, [((.25 * move) if i % 2 else (1.25 * move), (-1.)**i, 1 if i % 2 else 4)] * 2) for i in range(6)]
mixed_training_data = training_data + [((-1)**i, np.array((1.25 * ((i + 1) // 2),)), (.25 * move) if i % 2 else (1.25 * move), result, [((.25 * move) if i % 2 else (1.25 * move), (-1.)**i, 2 if i % 2 else 8)]) for i in range(6)]
training_data *= 100
mixed_training_data *= 50
MLPModelConfig.hidden_size = 32

def test_policy_model_stats(monkeypatch):
    logs = []
    def log(data, *args, **kwargs):
        if len(data) > 1:
            logs.append(data)

    monkeypatch.setattr(wandb, "log", log)
    model = PolicyMCTSModel(
        game_spec
    )
    model.learn(training_data[:10], stub_game.get_symmetries)
    model.learn(mixed_training_data[:20], stub_game.get_symmetries)

    expected_keys = ["loss", "value_loss", "policy_loss", "entropy_loss", "entropy"]

    assert len(logs) > 0
    for data in logs:
        for key in expected_keys:
            assert key in data
            assert "val_" + key in data

def test_reinforce_model_stats(monkeypatch):
    logs = []
    def log(data, *args, **kwargs):
        if len(data) > 1:
            logs.append(data)

    monkeypatch.setattr(wandb, "log", log)
    model = ReinforceMCTSModel(
        game_spec
    )
    model.learn(training_data[:10], stub_game.get_symmetries)
    model.learn(mixed_training_data[:20], stub_game.get_symmetries)

    expected_keys = ["loss", "value_loss", "policy_loss", "entropy_loss", "entropy"]

    assert len(logs) > 0
    for data in logs:
        for key in expected_keys:
            assert key in data
            assert "val_" + key in data
        assert 0 <= data["clip_fraction"] and data["clip_fraction"] <= 1
        assert 0 <= data["val_clip_fraction"] and data["val_clip_fraction"] <= 1

def test_ppo_model_stats(monkeypatch):
    logs = []
    def log(data, *args, **kwargs):
        if len(data) > 1:
            logs.append(data)

    monkeypatch.setattr(wandb, "log", log)
    model = PPOMCTSModel(
        game_spec
    )
    model.learn(training_data[:10], stub_game.get_symmetries)
    model.learn(mixed_training_data[:20], stub_game.get_symmetries)

    expected_keys = ["loss", "value_loss", "policy_loss", "entropy_loss", "entropy"]

    assert len(logs) > 0
    for data in logs:
        for key in expected_keys:
            assert key in data
            assert "val_" + key in data
        assert 0 <= data["clip_fraction"] and data["clip_fraction"] <= 1
        assert 0 <= data["val_clip_fraction"] and data["val_clip_fraction"] <= 1

@mark.flaky
def test_policy_learns():
    model = ReinforceMCTSModel(
        game_spec,
        config=ReinforceMCTSModelConfig(
            vf_coeff=0.,
            ent_coeff=0.,
        ),
        model_factory=MLPModelFactory
    )

    model.learn(training_data, stub_game.get_symmetries)
          
    distribution = model.generate_distribution(training_data[0][1])
    assert tf.reduce_prod(distribution.prob(move * 1.25)) > 5 *tf.reduce_prod(distribution.prob(move * .25))

@mark.flaky
def test_value_learns():
    model = ReinforceMCTSModel(
        game_spec,
        config=ReinforceMCTSModelConfig(
            vf_coeff=1000,
            ent_coeff=0.,
        ),
        model_factory=MLPModelFactory
    )

    model.learn(training_data, stub_game.no_symmetries)

    assert np.abs(model.predict_values(training_data[0][1]) - result) < stub_game.max_move

@mark.flaky
def test_entropy_increases():
    model = ReinforceMCTSModel(
        game_spec,
        config=ReinforceMCTSModelConfig(
            vf_coeff=0.,
            ent_coeff=1000.,
        ),
        model_factory=MLPModelFactory
    )

    entropy = model.generate_distribution(training_data[0][1]).entropy()
    model.learn(training_data, stub_game.no_symmetries)
    assert tf.reduce_all(model.generate_distribution(training_data[0][1]).entropy() > entropy)

@mark.flaky
def test_policy_model_losses_converge():
    model = PolicyMCTSModel(
        game_spec,
        config=PolicyMCTSModelConfig(
            vf_coeff=.5,
            ent_coeff=0.,
        ),
        model_factory=MLPModelFactory
    )

    model.learn(
        training_data,
        stub_game.no_symmetries,
        TrainingConfig(
            training_epochs=20,
            batch_size=32,
        )
    )

    distribution = model.generate_distribution(training_data[0][1])
    assert np.abs(model.predict_values(training_data[0][1]) - result) < stub_game.max_move
    ratio = tf.reduce_prod(distribution.prob(move * 1.25)) / tf.reduce_prod(distribution.prob(move * .25))
    assert np.allclose(ratio, 5, atol=2.)

@mark.flaky
def test_reinforce_model_losses_converge():
    model = ReinforceMCTSModel(
        game_spec,
        config=ReinforceMCTSModelConfig(
            vf_coeff=1.,
            ent_coeff=0.,
        ),
        model_factory=MLPModelFactory
    )

    model.learn(
        training_data,
        stub_game.no_symmetries,
    )

    distribution = model.generate_distribution(training_data[0][1])
    assert np.abs(model.predict_values(training_data[0][1]) - result) < stub_game.max_move
    assert tf.reduce_prod(distribution.prob(move * 1.25)) > 5 *tf.reduce_prod(distribution.prob(move * .25))

@mark.slow
@mark.flaky
def test_model_learns_from_multiple_actions():
    max_move = StubGame.max_move
    StubGame.max_move = 1.
    game = StubGame(2)
    StubGame.max_move = max_move

    game.reset(0)
    move = np.ones(game_spec.move_spec.shape) / 10
    training_data = [(
        1, game.get_observation(), 7 * move, 1., [
            (move * 3, -1., 1),
            (move * 5, 0., 2),
            (move * 7, 2., 3)
        ],
    )]
    game.step(move * 7)
    training_data.append((
        0, game.get_observation(), 6 * move, 1., [
            (move * 4, -1., 2),
            (move * 6, 2., 4)
        ],
    ))
    training_data *= 100

    model = ReinforceMCTSModel(
        game_spec,
        config=ReinforceMCTSModelConfig(
            vf_coeff=.5,
            ent_coeff=0.
        ),
        model_factory=MLPModelFactory
    )
    model.learn(
        training_data,
        stub_game.no_symmetries
    )

    distribution = model.generate_distribution(game.get_observation())
    assert tf.reduce_prod(distribution.prob(move * 6.5)) > 5 *tf.reduce_prod(distribution.prob(move * 3.5))
    game.reset(0)
    distribution = model.generate_distribution(game.get_observation())
    assert tf.reduce_prod(distribution.prob(move * 6.5)) > 5 *tf.reduce_prod(distribution.prob(move * 3.5))

@mark.flaky
def test_ppo_model_losses_converge():
    model = PPOMCTSModel(
        game_spec,
        config=PPOMCTSModelConfig(
            vf_coeff=1.,
            ent_coeff=0.,
            clip_range=.5,
            target_kl=None
        ),
        model_factory=MLPModelFactory
    )

    for _ in range(4):
        model.learn(
            training_data,
            stub_game.no_symmetries,
            training_config=TrainingConfig(
                training_epochs=4,
            )
        )

    distribution = model.generate_distribution(training_data[0][1])
    assert np.abs(model.predict_values(training_data[0][1]) - result) < stub_game.max_move
    assert tf.reduce_prod(distribution.prob(move * 1.25)) > 5 *tf.reduce_prod(distribution.prob(move * .25))

def test_ppo_std_changes():
    model = PPOMCTSModel(
        game_spec,
        config=PPOMCTSModelConfig(
            vf_coeff=1.,
            ent_coeff=0.,
            clip_range=.5,
            target_kl=None
        ),
        model_factory=MLPModelFactory
    )

    initial_stddev = model.generate_distribution(training_data[0][1]).stddev().numpy()
    assert np.allclose(initial_stddev, 1.)

    model.learn(
        training_data,
        stub_game.no_symmetries,
        training_config=TrainingConfig(
            training_epochs=4,
        )
    )
    
    stddev = model.generate_distribution(training_data[0][1]).stddev().numpy()
    assert not np.allclose(stddev, initial_stddev)
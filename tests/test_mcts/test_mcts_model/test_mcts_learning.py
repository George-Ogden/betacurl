import tensorflow as tf
import numpy as np

from dm_env.specs import BoundedArray
from pytest import mark
import wandb

from src.distribution import CombDistributionConfig, CombDistributionFactory, DistributionConfig
from src.mcts import PolicyMCTSModel, PolicyMCTSModelConfig, PPOMCTSModel, PPOMCTSModelConfig
from src.model import MLPModelConfig, MLPModelFactory, TrainingConfig
from src.game import Game, GameSpec

from tests.utils import MDPStubGame

MLPModelConfig.hidden_size = 32

max_move = MDPStubGame.max_move
MDPStubGame.max_move = 1.5
stub_game = MDPStubGame(6)
stub_game.max_move = MDPStubGame.max_move
MDPStubGame.max_move = max_move
game_spec = stub_game.game_spec
move = np.ones(game_spec.move_spec.shape)

result = 3.
training_data = [((-1)**i, np.array((1.25 * ((i + 1) // 2), 0)), (.25 * move) if i % 2 else (1.25 * move), result, [((.25 * move) if i % 2 else (1.25 * move), (-1.)**i, 1 if i % 2 else 4)] * 2) for i in range(6)]
mixed_training_data = training_data + [((-1)**i, np.array((1.25 * ((i + 1) // 2), 0)), (.25 * move) if i % 2 else (1.25 * move), result, [((.25 * move) if i % 2 else (1.25 * move), (-1.)**i, 2 if i % 2 else 8)]) for i in range(6)]
training_data *= 100
mixed_training_data *= 50

training_config = TrainingConfig(
    batch_size=32,
    training_epochs=5,
    lr=1e-2
)

def test_policy_model_stats(monkeypatch):
    logs = []
    def log(data, *args, **kwargs):
        if len(data) > 1:
            logs.append(data)

    monkeypatch.setattr(wandb, "log", log)
    model = PolicyMCTSModel(
        game_spec
    )
    model.learn(training_data[:10], stub_game.get_symmetries, training_config=training_config)
    model.learn(mixed_training_data[:20], stub_game.get_symmetries, training_config=training_config)

    expected_keys = ["loss", "value_loss", "policy_loss", "entropy_loss", "entropy"]

    assert len(logs) > 0
    for data in logs:
        for key in expected_keys:
            assert key in data
            assert "val_" + key in data

def test_ppo_model_stats(monkeypatch):
    logs = []
    def log(data, *args, **kwargs):
        if len(data) > 1:
            logs.append(data)

    monkeypatch.setattr(wandb, "log", log)
    model = PPOMCTSModel(
        game_spec
    )
    model.learn(training_data[:10], stub_game.get_symmetries, training_config=training_config)
    model.learn(mixed_training_data[:20], stub_game.get_symmetries, training_config=training_config)

    expected_keys = ["loss", "value_loss", "policy_loss", "entropy_loss", "entropy"]

    assert len(logs) > 0
    for data in logs:
        for key in expected_keys:
            assert key in data
            assert "val_" + key in data
        assert 0 <= data["clip_fraction"] and data["clip_fraction"] <= 1
        assert 0 <= data["val_clip_fraction"] and data["val_clip_fraction"] <= 1

@mark.flaky
def test_policy_learns_correct_comb():
    game_spec = GameSpec(
        move_spec=BoundedArray(
            shape=(1,),
            dtype=np.float32,
            minimum=-1.,
            maximum=1.
        ),
        observation_spec=BoundedArray(
            shape=(1,),
            dtype=np.float32,
            minimum=-1.,
            maximum=1.
        ),
        value_spec=BoundedArray(
            shape=(),
            dtype=np.float32,
            minimum=-1.,
            maximum=1.
        )
    )
    training_data = [
        (1, np.zeros(1), np.array((-0,)), 0., [(np.array((0,)), 0., 1), (np.array((.5,)), 0., 3), (np.array((.75,)), 0., 1.)])
    ]
    predicted_distribution = [0., 0., .2, .7, .1] # [-1, -.5, 0, .5, 1]
    model = PolicyMCTSModel(
        game_spec,
        model_factory=MLPModelFactory,
        config=PolicyMCTSModelConfig(
            vf_coeff=0.,
            ent_coeff=0.,
            distribution_config=CombDistributionConfig(
                granularity=5,
                noise_ratio=0.
            ),
        ),
        DistributionFactory=CombDistributionFactory
    )
    model.learn(training_data * 100, Game.no_symmetries, training_config=training_config)
    distribution = model.generate_distribution(training_data[0][1])
    assert np.allclose(distribution.pdf, predicted_distribution, atol=1e-1)

@mark.flaky
def test_policy_learns_correct_comb_multiple_actions():
    game_spec = GameSpec(
        move_spec=BoundedArray(
            shape=(2,),
            dtype=np.float32,
            minimum=(-1., 0.),
            maximum=(1., 2.)
        ),
        observation_spec=BoundedArray(
            shape=(1,),
            dtype=np.float32,
            minimum=-1.,
            maximum=1.
        ),
        value_spec=BoundedArray(
            shape=(),
            dtype=np.float32,
            minimum=-1.,
            maximum=1.
        )
    )
    training_data = [
        (1, np.zeros(1), np.array((-1,0.)), 0., [(np.array((0,.5)), 0., 1), (np.array((.5,0.)), 0., 3), (np.array((.75,1.)), 0., 1.)])
    ]
    predicted_distribution = [
        [0., 0., .2, .7, .1],
        [.6, .2, .2, .0, .0],
    ]
    model = PolicyMCTSModel(
        game_spec,
        model_factory=MLPModelFactory,
        config=PolicyMCTSModelConfig(
            vf_coeff=0.,
            ent_coeff=0.,
            distribution_config=CombDistributionConfig(
                noise_ratio=0.,
                granularity=5,
            ),
        ),
        DistributionFactory=CombDistributionFactory
    )
    model.learn(training_data * 100, Game.no_symmetries, training_config=training_config)
    distribution = model.generate_distribution(training_data[0][1])
    assert np.allclose(distribution.pdf, predicted_distribution, atol=1e-1)

@mark.flaky
def test_value_learns():
    model = PolicyMCTSModel(
        game_spec,
        config=PolicyMCTSModelConfig(
            vf_coeff=1000,
            ent_coeff=0.,
        ),
        model_factory=MLPModelFactory
    )

    model.learn(training_data, stub_game.no_symmetries, training_config=training_config)

    assert np.abs(model.predict_values(training_data[0][1]) - result) < stub_game.max_move

@mark.flaky
def test_entropy_increases():
    model = PolicyMCTSModel(
        game_spec,
        config=PolicyMCTSModelConfig(
            vf_coeff=0.,
            ent_coeff=1000.,
        ),
        model_factory=MLPModelFactory
    )

    entropy = model.generate_distribution(training_data[0][1]).entropy()
    model.learn(training_data, stub_game.no_symmetries, training_config=training_config)
    assert tf.reduce_sum(model.generate_distribution(training_data[0][1]).entropy()) > tf.reduce_sum(entropy)

@mark.flaky
def test_policy_model_losses_converge():
    model = PolicyMCTSModel(
        game_spec,
        config=PolicyMCTSModelConfig(
            vf_coeff=.5,
            ent_coeff=0.,
            distribution_config=DistributionConfig(
                noise_ratio=0.,
            )
        ),
        model_factory=MLPModelFactory
    )

    model.learn(training_data, stub_game.no_symmetries, training_config=training_config)

    distribution = model.generate_distribution(training_data[0][1])
    assert np.abs(model.predict_values(training_data[0][1]) - result) < stub_game.max_move
    assert tf.reduce_prod(distribution.prob(move * 1.25)) > 10 * tf.reduce_prod(distribution.prob(move * 0))
    assert tf.reduce_prod(distribution.prob(move * .25)) > 10 * tf.reduce_prod(distribution.prob(move * 0))

@mark.slow
@mark.flaky
def test_model_learns_from_multiple_actions():
    max_move = MDPStubGame.max_move
    MDPStubGame.max_move = 1.
    game = MDPStubGame(2)
    MDPStubGame.max_move = max_move

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

    model = PPOMCTSModel(
        game_spec,
        config=PPOMCTSModelConfig(
            vf_coeff=.5,
            ent_coeff=0.
        ),
        model_factory=MLPModelFactory
    )
    model.learn(training_data, stub_game.no_symmetries, training_config=training_config)

@mark.slow
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
        model.learn(training_data, stub_game.no_symmetries, training_config=training_config)

    distribution = model.generate_distribution(training_data[0][1])
    assert np.abs(model.predict_values(training_data[0][1]) - result) < stub_game.max_move
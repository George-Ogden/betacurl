import tensorflow as tf
import numpy as np

from dm_env.specs import Array, BoundedArray
from pytest import mark

from src.mcts import ReinforceMCTSModel, ReinforceMCTSModelConfig
from src.model import DenseModelFactory
from src.game import GameSpec

from tests.utils import StubGame

game = StubGame(10)

game_spec = game.game_spec
move_spec = game_spec.move_spec
observation_spec = game_spec.observation_spec

model = ReinforceMCTSModel(
    game_spec=game_spec,
)

def test_value_network():
    observation = game.get_observation()
    value = model.predict_values(observation)
    assert value.ndim == 0

    observations = np.array([game.get_observation() for _ in range(5)])
    value = model.predict_values(observations)
    assert value.shape == (5,)
    assert np.allclose(value, np.array(value).mean())

def test_policy_network():
    observation = game.get_observation()
    distribution = model.generate_distribution(observation)
    for i in range(1000):
        game.validate_action(distribution.sample().numpy())

@mark.flaky
def test_features_are_reasonable():
    pseudo_observations = np.random.uniform(
        low=observation_spec.minimum,
        high=observation_spec.maximum,
        size=(1000,) + observation_spec.shape
    )
    features = model.feature_extractor(pseudo_observations)
    assert np.abs(np.mean(features)) < 1.
    assert np.std(features) < 1.

def test_config_is_used():
    model = ReinforceMCTSModel(
        game_spec=game_spec,
        model_factory=DenseModelFactory,
        config=ReinforceMCTSModelConfig(
            feature_size=10,
            vf_coeff=.5,
            ent_coeff=.1,
            max_grad_norm=1.,
            clip_range=1.5
        )
    )

    assert model.feature_size == 10
    features = model.feature_extractor(np.random.randn(20, 1))
    assert features.shape == (20, 10)

    assert model.vf_coeff == .5
    assert model.ent_coeff == .1
    assert model.max_grad_norm == 1.
    assert model.clip_range == 1.5

def test_deterministic_outside_training():
    model = ReinforceMCTSModel(
        game_spec=game_spec
    )

    observation = np.ones_like(game.get_observation())
    features = model.feature_extractor(observation, training=False)
    features2 = model.feature_extractor(observation, training=False)
    assert tf.reduce_all(features == features2)

    value = model.predict_values(observation, training=False)
    value2 = model.predict_values(observation, training=False)
    assert tf.reduce_all(value == value2)

    dist = model.generate_distribution(observation, training=False)
    dist2 = model.generate_distribution(observation, training=False)

    assert np.allclose(dist.kl_divergence(dist2), 0.)

def test_training_transform():
    game = StubGame(2)
    game.reset(0)
    move = np.ones(move_spec.shape)
    training_data = [(
        1, game.get_observation(), 7 * move, 1., [
            (move * 3, 3, .2),
            (move * 5, 5, .3),
            (move * 7, 7, .5)
        ],
    )]
    game.step(move * 7)
    training_data.append((
        0, game.get_observation(), 6 * move, 1., [
            (move * 4, 4, .4),
            (move * 6, 6, .6)
        ],
    ))

    class DummyMCTSModel(ReinforceMCTSModel):
        def fit(self, dataset, training_config):
            self.dataset = dataset

    model = DummyMCTSModel(game_spec)
    model.learn(
        training_data, game.get_symmetries
    )
    seen_actions = set()
    for observation, actions, values, advantages in model.dataset:
        assert (observation[0] % 2 == 0) ^ tf.reduce_all(actions % 2 == 0)
        for action, advantage in zip(actions, advantages):
            seen_actions.update(list(action.numpy()))
            assert tf.reduce_all(action == advantage)
        assert observation[0] == 0 or tf.reduce_all(tf.sign(observation)[0] == tf.sign(values))
    assert len(seen_actions) == 5

@mark.flaky
def test_without_bounds():
    game_spec = GameSpec(
        move_spec=BoundedArray(
            shape=(2,2),
            dtype=np.float32,
            minimum=np.zeros((2,2), dtype=np.float32),
            maximum=np.ones((2,2), dtype=np.float32),
        ),
        observation_spec=Array(
            shape=(3,),
            dtype=np.float32
        )
    )
    model = ReinforceMCTSModel(
        game_spec=game_spec
    )
    observations = np.random.randn(100, 3)
    moves = model.generate_distribution(observation=observations)
    assert tf.reduce_mean(moves.mean()) < 2.
    values = model.predict_values(observations)
    assert values.shape == (100, )
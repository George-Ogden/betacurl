import tensorflow as tf
import numpy as np

from pytest import mark

from src.model import DenseModelFactory
from src.mcts import MCTSModel, MCTSModelConfig, SamplingMCTSModel

from tests.utils import StubGame

game = StubGame(10)

game_spec = game.game_spec
move_spec = game_spec.move_spec
observation_spec = game_spec.observation_spec

model = MCTSModel(
    game_spec=game_spec,
)
sampling_model = SamplingMCTSModel(
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

def test_default_scaling_spec():
    model = MCTSModel(
        game_spec=game_spec
    )
    mean = (move_spec.minimum + move_spec.maximum) / 2
    scaling_offset = model.policy_head.layers[-1].offset
    assert np.allclose(scaling_offset[:,0], mean)

def test_half_specified_scaling_spec():
    model = MCTSModel(
        game_spec=game_spec,
        scaling_spec=move_spec.maximum
    )
    scaling_offset = model.policy_head.layers[-1].offset
    assert np.allclose(scaling_offset[:,0], move_spec.maximum)

def test_fully_specified_scaling_spec():
    model = MCTSModel(
        game_spec=game_spec,
        scaling_spec=np.stack(
            (
                move_spec.minimum,
                move_spec.maximum
            ),
            axis=-1
        )
    )
    scaling_offset = model.policy_head.layers[-1].offset
    assert np.allclose(scaling_offset[:,0], move_spec.minimum)
    assert np.allclose(scaling_offset[:,1], np.log(move_spec.maximum))

@mark.probabilistic
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
    model = MCTSModel(
        game_spec=game_spec,
        model_factory=DenseModelFactory,
        config=MCTSModelConfig(
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
    model = MCTSModel(
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

    assert tf.reduce_all(dist.loc == dist2.loc)
    assert tf.reduce_all(dist.scale == dist2.scale)

def test_non_deterministic_during_training():
    observation = np.ones_like(game.get_observation())

    features = model.feature_extractor(observation, training=True)
    features2 = model.feature_extractor(observation, training=True)
    assert not tf.reduce_all(features == features2)

    value = model.predict_values(observation, training=True)
    value2 = model.predict_values(observation, training=True)
    assert not tf.reduce_all(value == value2)

    dist = model.generate_distribution(observation, training=True)
    dist2 = model.generate_distribution(observation, training=True)
    assert not tf.reduce_all(dist.loc == dist2.loc)
    assert not tf.reduce_all(dist.scale == dist2.scale)

def test_action_value_predictions():
    observation = np.random.randn(*observation_spec.shape)

    values = sampling_model.predict_values(observation, training=False)

    actions = [np.random.rand(*move_spec.shape) for _ in range(5)]
    assert tf.reduce_all(
            sampling_model.predict_action_values(
                observation,
                np.array(actions)
            ) != values
        )

def test_action_value_predictions_no_change():
    observation = np.random.randn(*observation_spec.shape)

    values = sampling_model.predict_values(observation, training=False)

    observation_head = sampling_model.observation_head
    sampling_model.observation_head = lambda x, training=False: x[0]
    actions = [np.random.rand(*move_spec.shape) for _ in range(5)]
    assert np.allclose(
        sampling_model.predict_action_values(
            observation,
            np.array(actions)
        ),
        values
    )


    # reset head
    sampling_model.observation_head = observation_head
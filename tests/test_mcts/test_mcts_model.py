import numpy as np

from pytest import mark

from src.model import DenseModelFactory
from src.mcts import MCTSModel, MCTSModelConfig

from tests.utils import StubGame

game = StubGame(10)

game_spec = game.game_spec
move_spec = game_spec.move_spec
observation_spec = game_spec.observation_spec

model = MCTSModel(
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
            max_grad_norm=1.
        )
    )

    assert model.feature_size == 10
    features = model.feature_extractor(np.random.randn(20, 1))
    assert features.shape == (20, 10)

    assert model.vf_coeff == .5
    assert model.ent_coeff == .1
    assert model.max_grad_norm == 1.
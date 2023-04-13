import tensorflow as tf
import numpy as np

from pytest import mark

from src.mcts.model.fourier import FourierDistribution
from src.mcts import FourierMCTSModel, FourierMCTSModelConfig

from tests.utils import MDPStubGame

test_distribution = FourierDistribution(
    coefficients = tf.reshape(tf.range(24, dtype=tf.float32), (4, 3, 2)),
    range = tf.constant([2., 4.]),
)

action_size = MDPStubGame.action_size
MDPStubGame.action_size = 1
game = MDPStubGame(6)
MDPStubGame.action_size = action_size

game_spec = game.game_spec
observation_spec = game_spec.observation_spec
move_spec = game_spec.move_spec

def test_distribution_pdf_cdf():
    assert tf.reduce_all(test_distribution.pdf >= 0)
    assert tf.reduce_all(0 <= test_distribution.cdf) and tf.reduce_all(test_distribution.cdf <= 1+1e-2)
    assert np.allclose(test_distribution.cdf[:, 0], 0., atol=1e-2)
    assert np.allclose(test_distribution.cdf[:,-1], 1., atol=1e-2)

def test_distribution_stats():
    mean = test_distribution.mean()
    mode = test_distribution.mode()
    std = test_distribution.stddev()
    variance = test_distribution.variance()

    assert mean.shape == (4,)
    assert mode.shape == (4,)
    assert std.shape == (4,)
    assert variance.shape == (4,)

    assert tf.reduce_all(2 <= mean) and tf.reduce_all(mean <= 4)
    assert tf.reduce_all(2 <= mode) and tf.reduce_all(mode <= 4)
    assert tf.reduce_all(0 < variance) and tf.reduce_all(variance < 1)
    assert np.allclose(std, tf.sqrt(variance))

def test_distribution_sample():
    for sample in test_distribution.sample(1000):
        assert sample.shape == (4,)
        assert tf.reduce_all(2 <= sample) and tf.reduce_all(sample <= 4)
        assert tf.reduce_all(test_distribution.prob(sample) >= 0)

def test_config_is_used():
    model = FourierMCTSModel(
        game_spec=game_spec,
        config=FourierMCTSModelConfig(
            fourier_features=7,
            feature_size=32
        )
    )

    assert np.prod(model.policy_head(np.random.rand(1, 32)).shape) % 7 == 0

def test_distribution_generation():
    model = FourierMCTSModel(game_spec=game_spec)
    distribution = model.generate_distribution(game.reset().observation)
    assert distribution.sample().shape == ()
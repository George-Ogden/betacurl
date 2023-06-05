import tensorflow as tf
import numpy as np

from pytest import mark

from src.mcts import PolicyMCTSModel, PolicyMCTSModelConfig
from src.mcts.model.comb import CombDistribution

from tests.utils import MDPStubGame

pdf = np.linspace(0, 2, 100, dtype=np.float32) / 100
test_distribution = CombDistribution(
    pdf = tf.constant(np.tile(pdf, (4, 1))),
    bounds = tf.constant([[2., 4.] for _ in range(4)]),
)
test_multi_distribution = CombDistribution(
    pdf = tf.constant(np.tile(pdf, (5, 4, 1))),
    bounds = tf.constant([[[2., 4.] for _ in range(4)] for _ in range(5)]),
)

def test_distribution_pdf_cdf():
    assert tf.reduce_all(test_distribution.pdf >= 0)
    assert tf.reduce_all(0 <= test_distribution._cdf) and tf.reduce_all(test_distribution._cdf <= 1+1e-2)
    assert np.allclose(test_distribution._cdf[:, 0], 0., atol=1e-2)
    assert np.allclose(test_distribution._cdf[:,-1], 1., atol=1e-2)

    assert tf.reduce_all(test_multi_distribution.pdf >= 0)
    assert tf.reduce_all(0 <= test_multi_distribution._cdf) and tf.reduce_all(test_multi_distribution._cdf <= 1+1e-2)
    assert np.allclose(test_multi_distribution._cdf[:, 0], 0., atol=1e-2)
    assert np.allclose(test_multi_distribution._cdf[:,-1], 1., atol=1e-2)

def test_distribution_stats():
    mean = test_distribution.mean()
    mode = test_distribution.mode()
    std = test_distribution.stddev()
    variance = test_distribution.variance()

    assert mean.shape == (4,)
    assert mode.shape == (4,)
    assert std.shape == (4,)
    assert variance.shape == (4,)

    assert np.allclose(mean, 3.33, atol=5e-2)
    assert np.allclose(mode, 4.00, atol=5e-2)
    assert tf.reduce_all(0 < variance)
    assert np.allclose(std, tf.sqrt(variance))

def test_multi_distribution_stats():
    mean = test_multi_distribution.mean()
    mode = test_multi_distribution.mode()
    std = test_multi_distribution.stddev()
    variance = test_multi_distribution.variance()

    assert mean.shape == (5,4)
    assert mode.shape == (5,4)
    assert std.shape == (5,4)
    assert variance.shape == (5,4)

    assert np.allclose(mean, 3.33, atol=5e-2)
    assert np.allclose(mode, 4.00, atol=5e-2)
    assert tf.reduce_all(0 < variance)
    assert np.allclose(std, tf.sqrt(variance))

def test_distribution_sample():
    samples = test_distribution.sample(100)
    assert samples.shape == (100, 4)
    assert tf.reduce_all(2 <= samples) and tf.reduce_all(samples <= 4)
    probs = test_distribution.prob(samples)
    assert probs.shape == (100, 4)
    assert tf.reduce_all(probs >= 0)

def test_multi_distribution_sample():
    samples = test_multi_distribution.sample(100)
    assert samples.shape == (100, 5, 4)
    assert tf.reduce_all(2 <= samples) and tf.reduce_all(samples <= 4)
    probs = test_multi_distribution.prob(samples)
    assert probs.shape == (100, 5, 4)
    assert tf.reduce_all(probs >= 0)

@mark.flaky
def test_distribution_correlations():
    samples = test_distribution.sample(1000).numpy()
    correlations = np.corrcoef(samples, rowvar=False) - np.eye(4)
    assert np.allclose(correlations, 0., atol=.8)

    samples = test_multi_distribution.sample(1000).numpy()
    correlations = np.corrcoef(samples.reshape(1000, 20), rowvar=False) - np.eye(20)
    assert np.allclose(correlations, 0., atol=.8)

    correlations = np.corrcoef(samples.reshape(5000, 4), rowvar=False) - np.eye(4)
    assert np.allclose(correlations, 0., atol=.8)

def test_dirichlet_noise():
    game = MDPStubGame()
    model = PolicyMCTSModel(
        game_spec=game.game_spec,
        config = PolicyMCTSModelConfig(
            exploration_coefficient=1.
        )
    )
    noise_tolerance = model.noise_ratio
    assert noise_tolerance > 0
    observation = game.get_observation()

    distribution1 = model.generate_distribution(observation)
    distribution2 = model.generate_distribution(observation)
    print(distribution1._pdf, distribution2._pdf)
    assert tf.reduce_all(distribution1.kl_divergence(distribution2) > 0)
    assert np.allclose(distribution1._pdf, distribution2._pdf, atol=noise_tolerance)
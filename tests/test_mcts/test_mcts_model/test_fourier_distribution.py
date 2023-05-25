import tensorflow as tf
import numpy as np

from pytest import mark

from src.mcts.model.fourier import FourierDistribution

from tests.utils import MDPStubGame

test_distribution = FourierDistribution(
    coefficients = tf.reshape(tf.range(24, dtype=tf.float32), (4, 3, 2)),
    bounds = tf.constant([[2., 4.] for _ in range(4)]),
)
test_multi_distribution = FourierDistribution(
    coefficients = tf.reshape(tf.range(120, dtype=tf.float32), (5, 4, 3, 2)),
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

    assert tf.reduce_all(2 <= mean) and tf.reduce_all(mean <= 4)
    assert tf.reduce_all(2 <= mode) and tf.reduce_all(mode <= 4)
    assert tf.reduce_all(0 < variance) and tf.reduce_all(variance < 1)
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

    assert tf.reduce_all(2 <= mean) and tf.reduce_all(mean <= 4)
    assert tf.reduce_all(2 <= mode) and tf.reduce_all(mode <= 4)
    assert tf.reduce_all(0 < variance) and tf.reduce_all(variance < 1)
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
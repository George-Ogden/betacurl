import tensorflow as tf
import numpy as np

from src.mcts.model.fourier import FourierDistribution
from src.mcts import FourierMCTSModel, FourierMCTSModelConfig

test_distribution = FourierDistribution(
    coefficients = tf.reshape(tf.range(24, dtype=tf.float32), (4, 3, 2)),
    range = tf.constant([2., 4.]),
)

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
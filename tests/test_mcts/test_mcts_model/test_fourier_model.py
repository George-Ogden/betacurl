import tensorflow as tf
import numpy as np

from src.mcts.model.fourier import FourierDistribution
from src.mcts import FourierMCTSModel, FourierMCTSModelConfig

test_distribution = FourierDistribution(
    coefficients = tf.reshape(tf.range(24, dtype=tf.float32), (4, 3, 2)),
    range = tf.constant([[2., 4.]]),
)

def test_distribution_pdf_cdf():
    assert tf.reduce_all(test_distribution.pdf >= 0)
    assert tf.reduce_all(0 <= test_distribution.cdf) and tf.reduce_all(test_distribution.cdf <= 1+1e-2)
    assert np.allclose(test_distribution.cdf[:, 0], 0., atol=1e-2)
    assert np.allclose(test_distribution.cdf[:,-1], 1., atol=1e-2)
import tensorflow as tf
import numpy as np

from pytest import mark

from src.mcts import PolicyMCTSModel

test_data = [
    (0, 0),
    (2, 0.734),
    (-2, -0.734),
    (100, 9.150),
    (-100, -9.150),
]

@mark.parametrize("value, expected", test_data)
def test_value_scaling(value, expected):
    assert np.allclose(
        PolicyMCTSModel.scale_values(tf.constant([value], dtype=tf.float32)).numpy(),
        expected,
        atol=1e-3
    )

@mark.parametrize("value, expected", test_data)
def test_value_inverse_scaling(value, expected):
    assert np.allclose(
        PolicyMCTSModel.inverse_scale_values(
            PolicyMCTSModel.scale_values(tf.constant([value], dtype=tf.float32))
        ).numpy(),
        value,
        atol=1e-3
    )
                                         
                                     
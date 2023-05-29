import tensorflow as tf
import numpy as np

from src.utils.support import support_to_value, value_to_support

def test_support_to_value():
    support = tf.constant([1, 2, 3], dtype=tf.float32)
    coefficients = tf.constant([[0.1, 0.3, 0.6], [0.4, 0.6, 0.]], dtype=tf.float32)
    expected_output = tf.constant([2.5, 1.6], dtype=tf.float32)

    output = support_to_value(coefficients, support)
    assert np.allclose(output, expected_output)

def test_value_to_support():
    support = tf.constant([1, 2, 3], dtype=tf.float32)
    values = tf.constant([2.5, 1.6], dtype=tf.float32)
    expected_output = tf.constant([[0, .5, .5], [.4, .6, 0.]], dtype=tf.float32)

    output = value_to_support(values, support)
    assert np.allclose(output, expected_output)
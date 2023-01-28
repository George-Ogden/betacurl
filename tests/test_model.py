from src.model import SimpleLinearModelFactory, SimpleLinearModelConfig
import tensorflow as tf
import numpy as np

config = SimpleLinearModelConfig(
    output_activation="sigmoid", hidden_size=8
)

def test_forward():
    model = SimpleLinearModelFactory.create_model(input_size=2, output_size=1, config=config)
    input = tf.random.normal((16, 2))
    output = model(input)
    assert output.shape == (16, 1)
    assert tf.reduce_all(output > 0)
    assert tf.reduce_all(output < 1)

def test_without_config():
    model = SimpleLinearModelFactory.create_model(input_size=2, output_size=1)
    input = tf.random.normal((16, 2))
    output = model(input)
    assert output.shape == (16, 1)
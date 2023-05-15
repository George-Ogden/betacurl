import tensorflow as tf
import numpy as np

from pytest import mark

from dm_env.specs import BoundedArray
from typing import List

from src.mcts import PolicyMCTSModel
from src.game import GameSpec

scaling_test_data = [
    (0, 0),
    (2, 0.734),
    (-2, -0.734),
    (100, 9.150),
    (-100, -9.150),
]

support_test_data = [
    (0, [0, 0, 0, 0, 1, 0, 0, 0, 0]),
    (2.7, [0, 0, 0, 0, 0, 0, .3, .7, 0]),
    (-2.7, [0, 0.7, 0.3, 0, 0, 0, 0, 0, 0]),
]

model = PolicyMCTSModel(
    game_spec=GameSpec(
        observation_spec=BoundedArray(
            shape=(1,),
            dtype=np.float32,
            minimum=-1,
            maximum=1,
        ),
        move_spec=BoundedArray(
            shape=(1,),
            dtype=np.float32,
            minimum=-1,
            maximum=1,
        ),
    )
)
model.value_coefficients = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4], dtype=np.float32)

@mark.parametrize("value, expected", scaling_test_data)
def test_value_scaling(value: float, expected: float):
    assert np.allclose(
        PolicyMCTSModel.scale_values(tf.constant([value], dtype=tf.float32)).numpy(),
        expected,
        atol=1e-3
    )

@mark.parametrize("value, expected", scaling_test_data)
def test_value_inverse_scaling(value: float, expected: float):
    assert np.allclose(
        PolicyMCTSModel.inverse_scale_values(
            PolicyMCTSModel.scale_values(tf.constant([value], dtype=tf.float32))
        ).numpy(),
        value,
        atol=1e-3
    )

@mark.parametrize("value, expected", support_test_data)                           
def test_value_to_logits(value: float, expected: List[float]):
    assert np.allclose(
        model.values_to_logits(tf.constant([value], dtype=tf.float32)).numpy(),
        np.array(expected),
        atol=1e-5
    )

@mark.parametrize("value, expected", support_test_data)
def test_logits_to_values(value: float, expected: List[float]):
    assert np.allclose(
        model.logits_to_values(
            model.values_to_logits(tf.constant([value], dtype=tf.float32))
        ).numpy(),
        value,
        atol=1e-5
    )
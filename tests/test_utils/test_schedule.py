import numpy as np

from src.utils import GeometricSchedule, LinearSchedule

def test_linear_schedule():
    schedule = LinearSchedule((0.11, 1.01), (11, 101))
    assert np.allclose(schedule[11], 0.11)
    assert np.allclose(schedule[55], 0.55)
    assert np.allclose(schedule[93], 0.93)
    assert np.allclose(schedule[101], 1.01)

    assert np.allclose(schedule[0.0], 0.11)
    assert np.allclose(schedule[0.2], 0.29)
    assert np.allclose(schedule[0.6], 0.65)
    assert np.allclose(schedule[1.0], 1.01)


def test_geommetric_schedule():
    schedule = GeometricSchedule((1e1, 1e6), (1, 6))
    assert np.allclose(schedule[1], 1e1)
    assert np.allclose(schedule[3], 1e3)
    assert np.allclose(schedule[6], 1e6)

    assert np.allclose(schedule[0.0], 1e1)
    assert np.allclose(schedule[0.2], 1e2)
    assert np.allclose(schedule[0.6], 1e4)
    assert np.allclose(schedule[1.0], 1e6)
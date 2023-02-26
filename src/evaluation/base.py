import numpy as np

from dm_env.specs import BoundedArray
from typing import Union

from ..utils import SaveableObject

class EvaluationStrategy(SaveableObject):
    DEFAULT_FILENAME = "evaluator.pickle"
    def __init__(self, observation_spec: BoundedArray):
        self.observation_shape = observation_spec.shape
        self.observation_range = np.stack((observation_spec.minimum, observation_spec.maximum), axis=0)

    def evaluate(self, observations: np.ndarray) -> Union[float, np.ndarray]:
        if observations.ndim == 1:
            return 0.
        else:
            return np.zeros(len(observations))
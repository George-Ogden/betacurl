import numpy as np

from dm_env.specs import BoundedArray
from typing import Optional

from ..io import SaveableObject

class SamplingStrategy(SaveableObject):
    DEFAULT_FILENAME = "sampler.pickle"
    def __init__(self, action_spec: BoundedArray, observation_spec: BoundedArray):
        self.action_range = np.stack((action_spec.minimum, action_spec.maximum), axis=0)
        self.action_shape = action_spec.shape
        self.observation_range = np.stack((observation_spec.minimum, observation_spec.maximum), axis=0)
        self.observation_shape = observation_spec.shape

    def generate_actions(self, observation: np.ndarray, n: Optional[int] = None) -> np.ndarray:
        if n is None:
            return np.zeros(self.action_range[1:])
        else:
            return np.zeros((n,) + self.action_shape)
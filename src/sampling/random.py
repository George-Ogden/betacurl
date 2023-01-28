from src.sampling.base import SamplingStrategy
import numpy as np

from typing import Optional

class RandomSamplingStrategy(SamplingStrategy):
    def generate_actions(self, observation: np.ndarray, n: Optional[int] = None) -> np.ndarray:
        if n is None:
            return np.random.uniform(low=self.action_range[0], high=self.action_range[1])
        else:
            return np.random.uniform(low=self.action_range[0], high=self.action_range[1], size=(n,) + self.action_shape)
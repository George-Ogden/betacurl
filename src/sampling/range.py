from src.sampling.base import SamplingStrategy
import numpy as np

from typing import Optional

class MinSamplingStrategy(SamplingStrategy):
    def generate_actions(self, observation: np.ndarray, n: Optional[int] = None) -> np.ndarray:
        if n is None:
            return self.action_range[0].copy()
        else:
            return np.tile(self.action_range[0], (n,) + (1,) * (self.action_range.ndim - 1))

class MaxSamplingStrategy(SamplingStrategy):
    def generate_actions(self, observation: np.ndarray, n: Optional[int] = None) -> np.ndarray:
        if n is None:
            return self.action_range[1].copy()
        else:
            return np.tile(self.action_range[1], (n,) + (1,) * (self.action_range.ndim - 1))
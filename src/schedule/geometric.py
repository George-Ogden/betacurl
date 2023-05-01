import numpy as np

from .base import Schedule
from typing import Tuple

class GeometricSchedule(Schedule):
    def __init__(self, values: Tuple[float, float], range: Tuple[int, int]) -> None:
        super().__init__(values, range)
        assert self.end_value > 0 and self.start_value > 0, "min and max values must be positive"
        self.start_log = np.log(self.start_value)
        self.end_log = np.log(self.end_value)

    def interpolate(self, t: float) -> float:
        return np.exp(self.start_log + (self.end_log - self.start_log) * t)
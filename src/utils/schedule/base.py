from abc import ABC, abstractmethod
from typing import Tuple, Union

class Schedule(ABC):
    def __init__(self, values: Tuple[float, float], range: Tuple[int, int]) -> None:
        super().__init__()
        self.start_value, self.end_value = values
        self.start_iteration, self.end_iteration = range
    
    @abstractmethod
    def interpolate(self, t: float) -> float:
        ...

    def __getitem__(self, iteration: Union[int, float]) -> float:
        if isinstance(iteration, int):
            iteration = (iteration - self.start_iteration) / ((self.end_iteration - self.start_iteration) or 1.)
        assert 0 <= iteration and iteration <= 1, f"iteration out of range"
        return self.interpolate(iteration)
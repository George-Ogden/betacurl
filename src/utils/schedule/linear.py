from .base import Schedule

class LinearSchedule(Schedule):
    def interpolate(self, t: float) -> float:
        return self.start_value + (self.end_value - self.start_value) * t
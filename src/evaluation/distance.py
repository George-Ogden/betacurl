from .base import EvaluationStrategy

import numpy as np

class DistanceEvaluationStrategy(EvaluationStrategy):
    def evaluate(self, observations: np.ndarray) -> float:
        batched_throughput = True
        if observations.ndim != 2:
            observations = np.expand_dims(observations, 0)
            batched_throughput = False
        evaluations = -np.linalg.norm(observations[:,:2], axis=-1) - np.linalg.norm(observations[:,2:], axis=-1)
        if not batched_throughput:
            evaluations = np.squeeze(evaluations, 0)
        return evaluations
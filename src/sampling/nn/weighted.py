from copy import deepcopy
import numpy as np

from typing import Callable, List, Tuple

from ...model import TrainingConfig

from .nn import NNSamplingStrategy

class WeightedNNSamplingStrategy(NNSamplingStrategy):
    def learn(
        self,
        training_history: List[Tuple[int, np.ndarray, np.ndarray, float]],
        augmentation_function: Callable[[np.ndarray, np.ndarray, float], List[Tuple[np.ndarray, np.ndarray, float]]],
        training_config: TrainingConfig = TrainingConfig()
    ):
        training_config = deepcopy(training_config)
        training_data = [(augmented_observation, augmented_action, reward * np.sign(player) * np.sign(reward)) for (player, observation, action, reward) in training_history for (augmented_observation, augmented_action, augmented_reward) in (augmentation_function(observation, action, reward))]
        observations, actions, weights = zip(*training_data)

        observations = self.add_noise_to_observations(observations)

        # remove patience and validation splitting
        training_config.training_patience = 0
        training_config.validation_split = 0
        training_config.fit_kwargs = dict(
            sample_weight = np.array(weights)
        )
        self.fit(observations, actions, training_config)
from tensorflow_probability import distributions
from dm_env import StepType
import tensorflow as tf
import numpy as np

from typing import Dict, Optional, Tuple

from ..utils.io import SaveableObject
from ..game import Game

from .model import ReinforceMCTSModel
from .widening import WideningMCTS
from .config import NNMCTSConfig

class NNMCTS(WideningMCTS, SaveableObject):
    CONFIG_CLASS = NNMCTSConfig
    DEFAULT_FILENAME = "nn_mcts.pickle"
    SEPARATE_ATTRIBUTES = ["model"]
    def __init__(
        self,
        game: Game,
        model: Optional[ReinforceMCTSModel] = None,
        config: NNMCTSConfig = NNMCTSConfig()
    ):
        super().__init__(game, config=config, action_generator=self.generate_action)

        self.model = model
        self.max_depth = config.max_rollout_depth

        self.planned_actions: Dict[bytes, distributions.Distribution] = {}

    def rollout(self, game: Game) -> float:
        returns = 0.
        for _ in range(self.max_depth):
            action = game.get_random_move()
            timestep = game.step(action)
            returns += timestep.reward or 0.
            if timestep.step_type == StepType.LAST:
                break
        else:
            if self.model is None:
                returns += 0.
            else:
                returns += self.model.predict_values(game.get_observation())
        return returns

    def generate_action(self, observation: np.ndarray) -> Tuple[np.ndarray, float]:
        if self.model is None:
            return (self.game.get_random_move(), 1.)

        encoding = self.encode(observation)
        if encoding not in self.planned_actions:
            self.planned_actions[encoding] = self.model.generate_distribution(observation)
        distribution = self.planned_actions[encoding]
        action = distribution.sample()
        prob = tf.reduce_prod(distribution.prob(action))
        return (
            tf.clip_by_value(
                action,
                self.action_spec.minimum,
                self.action_spec.maximum
            ).numpy(),
            prob.numpy()
        )

    def save(self, directory: str):
        # don't save planned actions
        planned_actions = self.planned_actions
        self.planned_actions = {}
        super().save(directory)
        self.planned_actions = planned_actions
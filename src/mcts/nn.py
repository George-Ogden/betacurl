from tensorflow_probability import distributions
import tensorflow as tf
from copy import copy
import numpy as np

from typing import Dict, Optional, Tuple
from enum import IntEnum, auto

from ..utils import SaveableObject
from ..game import Game

from .config import NNMCTSConfig
from .widening import WideningMCTS
from .fixed import FixedMCTS
from .model import MCTSModel
from .base import MCTS

class NNMCTSMode(IntEnum):
    FIXED: int = auto()
    WIDENING: int = auto()

class NNMCTS(MCTS, SaveableObject):
    CONFIG_CLASS = NNMCTSConfig
    DEFAULT_FILENAME = "nn_mcts.pickle"
    SEPARATE_ATTRIBUTES = ["model"]
    def __init__(
        self,
        game: Game,
        model: Optional[MCTSModel] = None,
        config: NNMCTSConfig = NNMCTSConfig(),
        initial_mode: NNMCTSMode = NNMCTSMode.WIDENING
    ):
        super().__init__(game, config=config)
        self.model = model
        # save config
        self.config = copy(config)
        self.max_depth = config.max_rollout_depth
        self.num_actions = config.num_actions
        self.cpw = config.cpw
        self.kappa = config.kappa

        self.planned_actions: Dict[bytes, distributions.Distribution] = {}
        self.set_mode(initial_mode)
    
    def set_mode(self, mode: NNMCTSMode):
        self.mode = mode
    
    def fix(self):
        self.set_mode(NNMCTSMode.FIXED)
    
    def widen(self):
        self.set_mode(NNMCTSMode.WIDENING)

    # define separate methods as Python mro is broken
    def rollout(self, game: Game) -> float:
        returns = 0.
        for _ in range(self.max_depth):
            action = game.get_random_move()
            timestep = game.step(action)
            returns += timestep.reward or 0.
            if timestep.step_type.last():
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
        # parents do not define custom saving
        super().save(directory)
        self.planned_actions = planned_actions
    
    def select_action(self, observation: np.ndarray) -> np.ndarray:
        match self.mode:
            case NNMCTSMode.FIXED:
                return FixedMCTS.select_action(self, observation)
            case NNMCTSMode.WIDENING:
                return WideningMCTS.select_action(self, observation)
            case _:
                raise ValueError(f"Unknown mode: {self.mode}")
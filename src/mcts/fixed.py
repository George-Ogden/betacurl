import numpy as np

from typing import Callable, Optional, Tuple

from .config import FixedMCTSConfig
from .base import MCTS

class FixedMCTS(MCTS):
    CONFIG_CLASS = FixedMCTSConfig
    def __init__(self, game: "Game", action_generator: Optional[Callable[[np.ndarray], Tuple[np.ndarray, float]]] = None, config: FixedMCTSConfig = FixedMCTSConfig()):
        """
        Args:
            game (Game): game to search
            action_generator (Optional[Callable[[np.ndarray], Tuple[np.ndarray, float]]], optional): function to generate a tuple of an action and its probability from an observation. Defaults to None.
        """
        super().__init__(game, config)

        if not action_generator:
            action_generator = self._default_move_generator
        self.generate_action = action_generator

        self.num_actions = config.num_actions

    def select_action(self, observation: np.ndarray) -> np.ndarray:
        node = self.get_node(observation)
        if not hasattr(node, "potential_actions"):
            actions, probs = zip(*[self.generate_action(observation) for _ in range(self.num_actions)])
            probs = np.array(probs)
            order = np.argsort(-probs)
            # keep actions in order of probability as selected by policy
            node.potential_actions = np.array(actions)[order]
            node.action_probs = probs[order]

        if len(node.transitions) < len(node.potential_actions):
            # try the most likely untried action
            return node.potential_actions[len(node.transitions)]

        return self.select_puct_action(observation)
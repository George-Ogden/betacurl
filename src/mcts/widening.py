import numpy as np

from typing import Callable, Optional, Tuple

from src.game import Game

from .config import WideningMCTSConfig
from .base import MCTS

class WideningMCTS(MCTS):
    def __init__(self, game: Game, action_generator: Optional[Callable[[np.ndarray], Tuple[np.ndarray, float]]] = None, config: WideningMCTSConfig = WideningMCTSConfig()):
        """
        Args:
            game (Game): game to search
            move_generation (Optional[Callable[[np.ndarray], List[Tuple[np.ndarray, float]]]], optional): function to take in an observation and return a list of tuples of actions and probabilities. Defaults to None.
        """
        super().__init__(game, config)
        
        if not action_generator:
            action_generator = self._default_move_generator
        self.generate_action = action_generator

        self.cpw = config.cpw
        self.kappa = config.kappa

    def select_action(self, observation: np.ndarray) -> np.ndarray:
        node = self.get_node(observation)
        # m(s) = c_pw * n(s)^kappa
        num_actions = np.ceil(self.cpw * node.num_visits ** self.kappa)
        if num_actions > len(node.transitions):
            action, prob = self.generate_action(observation)
            if hasattr(node, "action_probs"):
                node.action_probs = np.concatenate((node.action_probs, (prob,)))
            else:
                node.action_probs = np.array((prob,))
            return action

        return self.select_puct_action(observation)
import numpy as np

from typing import Callable, Optional, Tuple

from src.game import Game

from .config import FixedMCTSConfig
from .base import MCTS

class FixedMCTS(MCTS):
    def __init__(self, game: Game, action_generator: Optional[Callable[[np.ndarray], Tuple[np.ndarray, float]]] = None, config: FixedMCTSConfig = FixedMCTSConfig):
        """
        Args:
            game (Game): game to search
            action_generator (Optional[Callable[[np.ndarray], Tuple[np.ndarray, float]]], optional): function to generate a tuple of an action and its probability from an observation. Defaults to None.
        """
        super().__init__(game, config)
        self.num_actions = config.num_actions
        if not action_generator:
            action_generator = self._default_move_generator
        self.generate_action = action_generator

    def select_action(self, observation: np.ndarray) -> np.ndarray:
        node = self.get_node(observation)
        if not hasattr(node, "potential_actions"):
            actions, probs = zip(*[self.generate_action(observation) for _ in range(self.num_actions)])
            probs = np.array(probs)
            order = np.argsort(-probs)
            # keep actions in order of probability as selected by policy
            node.potential_actions = np.array(actions)[order]
            node.action_probs = probs[order] / probs.sum()

        if len(node.transitions) < len(node.potential_actions):
            # try the most likely untried action
            return node.potential_actions[len(node.transitions)]

        actions = node.transitions.values()
        q_values = np.array([
            action.reward + (
                0.
                if action.termination else
                self.nodes[action.next_state].expected_return
            )
            for action in actions
        ])
        u_values = (
            node.action_probs
            * self.cpuct 
            * [
                np.sqrt(node.num_visits) / (1 + action.num_visits) 
                for action in actions
            ]
        )
        values = u_values + q_values * self.game.player_delta
        return [action.action for action in actions][values.argmax()]
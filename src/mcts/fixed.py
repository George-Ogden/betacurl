import numpy as np

from typing import Callable, List, Tuple, Union

from src.game import Game

from .base import MCTS

class FixedMCTS(MCTS):
    cpuct: np.floating = np.array(1.) # "theoretically equal to √2; in practice usually chosen empirically"
    def __init__(self, game: Game, move_generator: Union[Callable[[np.ndarray], List[Tuple[np.ndarray, float]]], int] = 10):
        """
        Args:
            game (Game): game to search
            move_generation (Optional[Callable[[np.ndarray], List[Tuple[np.ndarray, float]]]], optional): function to take in an observation and return a list of tuples of actions and probabilities. Defaults to None.
        """
        super().__init__(game)
        if isinstance(move_generator, int):
            move_generator = self._default_move_generator(move_generator)
        self.generate_moves = move_generator

    def _default_move_generator(self, num_moves: int) -> Callable[[np.ndarray], List[Tuple[np.ndarray, float]]]:
        def generate_moves(observation: np.ndarray) -> List[Tuple[np.ndarray, float]]:
            return [(self.game.get_random_move(), 1.) for _ in range(num_moves)]
        return generate_moves

    def select_action(self, observation: np.ndarray) -> np.ndarray:
        node = self.get_node(observation)
        if not hasattr(node, "potential_actions"):
            actions, probs = zip(*self.generate_moves(observation))
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
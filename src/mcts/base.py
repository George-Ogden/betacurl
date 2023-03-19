from dm_env import StepType
from copy import copy
import numpy as np

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from abc import ABCMeta, abstractmethod

from ..game import Game

from .config import MCTSConfig

import numpy as np

@dataclass
class Transition:
    action: np.ndarray
    next_state: bytes # assume deterministic transition
    reward: float = 0. # assume deterministic reward
    discount: float = 1.
    num_visits: int = 0
    termination: bool = False

@dataclass
class Node:
    game: Game # partially completed game
    total_return: float = 0
    num_visits: int = 1
    transitions: Dict[bytes, Transition] = field(default_factory=dict)
    @property
    def expected_return(self) -> float:
        return self.total_return / self.num_visits

    def get_transition(self, action: np.ndarray) -> Optional[Transition]:
        return self.transitions.get(MCTS.encode(action), None)

    def set_transition(self, action: np.ndarray, transition: Transition):
        self.transitions[MCTS.encode(action)] = transition

class MCTS(metaclass=ABCMeta):
    CONFIG_CLASS = MCTSConfig
    def __init__(self, game: Game, config: MCTSConfig = MCTSConfig()):
        self.game = game
        self.action_spec = game.game_spec.move_spec
        self.nodes: Dict[bytes, Node] = {}
        
        self.config = copy(config)
        self.cpuct = config.cpuct

    @staticmethod
    def encode(state: np.ndarray) -> bytes:
        return state.astype(np.float32).tobytes()

    @abstractmethod
    def select_action(self, observation: np.ndarray) -> np.ndarray:
        ...

    def _default_move_generator(self, observation: np.ndarray) -> Tuple[np.ndarray, float]:
        return (self.game.get_random_move(), 1.)

    def get_actions(self, observation: np.ndarray) -> List[Transition]:
        return list(self.nodes[self.encode(observation)].transitions.values())

    def _get_action_probs(self, game: Game, temperature: float) -> Tuple[np.ndarray, np.ndarray]:
        observation = game.get_observation()
        actions = np.array([action.action for action in self.get_actions(observation)])
        visits = np.array([action.num_visits for action in self.get_actions(observation)])
        if temperature == 0:
            probs = np.zeros(len(actions), dtype=float)
            potential_actions = np.argwhere(visits == visits.max()).reshape(-1)
            probs[np.random.choice(potential_actions)] = 1.
        else:
            probs = visits ** (1. / temperature)
        return actions, probs

    def get_action_probs(self, game: Optional[Game] = None, temperature: float = 1.) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple of (actions, probs) (probs sum to 1)
        """
        if not game:
            game = self.game
        actions, probs = self._get_action_probs(game, temperature)
        if probs.sum() != 0:
            probs /= probs.sum()
        return actions, probs

    def get_node(self, observation: np.ndarray) -> Optional[Node]:
        return self.nodes.get(self.encode(observation), None)

    def set_node(self, observation: np.ndarray, node: Node):
        self.nodes[self.encode(observation)] = node

    def rollout(self, game: Game) -> float:
        multiplier = 1.
        reward = 0.
        while multiplier > game.eps:
            action = game.get_random_move()
            timestep = game.step(action)
            reward += (timestep.reward or 0.) * multiplier
            if timestep.step_type == StepType.LAST:
                break
            multiplier *= timestep.discount or 1.
        return reward

    def search(self, game: Optional[Game] = None):
        if not game:
            game = self.game
        observation = game.get_observation()
        if not (node := self.get_node(observation)):
            # game will be modified in rollout
            returns = self.rollout(game.clone())
            node = Node(
                game=game,
                num_visits=1,
                total_return=returns
            )
            self.set_node(observation, node)
            return returns

        action = self.select_action(observation)
        transition = node.get_transition(action)
        if transition:
            next_state = transition.next_state
            # running the simulation is the expensive part
            returns = transition.reward + transition.discount * (
                self.search(self.nodes[next_state].game)
                if not transition.termination else 0
            )
        else:
            # only copied when stepping
            game = game.clone()
            timestep = game.step(action)
            transition = Transition(
                action=action,
                next_state=self.encode(timestep.observation),
                reward=timestep.reward or 0.,
                termination=timestep.step_type == StepType.LAST,
                discount=timestep.discount or 1.,
                num_visits=0
            )
            node.set_transition(action, transition)
            returns = transition.reward + (
                transition.discount
                or 1.
            ) * (
                self.search(game)
                if not transition.termination else 0
            )
        node.num_visits += 1
        transition.num_visits += 1
        node.total_return += returns
        return returns

    def select_puct_action(self, observation: np.ndarray) -> np.ndarray:
        node = self.get_node(observation)
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
            node.action_probs / node.action_probs.sum()
            * self.cpuct 
            * [
                np.sqrt(node.num_visits) / (1 + action.num_visits) 
                for action in actions
            ]
        )
        values = u_values + q_values * self.game.player_delta
        return [action.action for action in actions][values.argmax()]

    def cleanup(self, game: Optional[Game] = None):
        """delete unreachable nodes"""
        if game is None:
            game = self.game
        root = self.get_node(game.get_observation())
        root.reachable = True
        queue = [root]
        while queue:
            node = queue.pop(0)
            for transition in node.transitions.values():
                neighbor = self.nodes.get(transition.next_state, None)
                if neighbor and not getattr(neighbor, "reachable", False):
                    neighbor.reachable = True
                    queue.append(neighbor)

        for key, node in list(self.nodes.items()):
            if getattr(node, "reachable", False):
                del node.reachable
            else:
                del self.nodes[key]

    def freeze(self):
        """
        calculate values for actions
        this can only be done once the values no longer update
        """
        for node in self.nodes.values():
            if not node.transitions:
                continue

            visits = [transition.num_visits for transition in node.transitions.values()]
            mean = np.mean(visits)
            scale = max(np.std(visits), 1.)
            for transition in node.transitions.values():
                # rescale visits to perform REINFORCE with baseline
                transition.advantage = (transition.num_visits - mean) / scale
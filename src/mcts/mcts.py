from dm_env import StepType
from copy import deepcopy
import numpy as np

from dataclasses import dataclass, field
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Optional, Tuple

from src.game import Game

@dataclass
class Transition:
    action: np.ndarray
    next_state: bytes
    reward: float = 0 # assume deterministic reward
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

class MCTS(metaclass=ABCMeta):
    def __init__(self, game: Game):
        self.game = game
        self.action_spec = game.game_spec.move_spec
        self.nodes: Dict[bytes, Node] = {}

    @staticmethod
    def encode(state: np.ndarray) -> bytes:
        return state.tobytes()

    @abstractmethod
    def select_action(self, observation: np.ndarray) -> np.ndarray:
        ...
    
    def get_actions(self, observation: np.ndarray) -> List[Transition]:
        return list(self.nodes[self.encode(observation)].transitions.values())
    
    @abstractmethod
    def _get_action_probs(self, game: Game, temperature: float) -> Tuple[np.ndarray, np.ndarray]:
        ...

    def get_action_probs(self, game: Optional[Game] = None, temperature: float = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple of (actions, probs) (probs sub to 1)
        """
        if not game:
            game = self.game
        actions, probs = self._get_action_probs(game, temperature)
        probs = np.array(probs, dtype=float)
        if probs.sum() != 0:
            probs /= probs.sum()
        return np.array(actions), probs
    
    def rollout(self, game: Game):
        action = game.get_random_move()
        timestep = game.step(action)
        if timestep.step_type == StepType.LAST:
            return timestep.reward
        return (timestep.reward or 0) + self.rollout(game)

    def search(self, game: Optional[Game] = None):
        if not game:
            game = self.game
        observation = game.get_observation()
        state = self.encode(observation)
        if not state in self.nodes:
            # game will be modified in rollout
            returns = self.rollout(deepcopy(game))
            node_information = Node(
                game=game,
                num_visits=1,
                total_return=returns
            )
            self.nodes[state] = node_information
            return returns

        node_information = self.nodes[state]
        action = self.select_action(observation)
        action_representation = self.encode(action)
        if action_representation in node_information.transitions:
            action_information = node_information.transitions[action_representation]
            next_state = action_information.next_state
            # running the simulation is the expensive part
            returns = action_information.reward + (
                self.search(self.nodes[next_state].game)
                if not action_information.termination else 0
            )
        else:
            # only copied when stepping
            game = deepcopy(game)
            timestep = game.step(action)
            action_information = Transition(
                action=action,
                next_state=self.encode(timestep.observation),
                reward=timestep.reward,
                termination=timestep.step_type == StepType.LAST,
                num_visits=0
            )
            node_information.transitions[action_representation] = action_information
            returns = timestep.reward + (
                self.search(game) 
                if not action_information.termination else 0
            )
        node_information.num_visits += 1
        action_information.num_visits += 1
        node_information.total_return += returns
        return returns
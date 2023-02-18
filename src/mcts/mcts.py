from dm_env import StepType
import numpy as np

from dataclasses import dataclass, field
from abc import ABCMeta, abstractmethod
from typing import Dict, Optional
from copy import deepcopy

from src.game import Game

@dataclass
class ActionInformation:
    next_state: str
    reward: float = 0 # assume deterministic reward
    num_visits: int = 0
    termination: bool = False

@dataclass
class NodeInformation:
    game: Game # partially completed game
    total_return: float = 0
    num_visits: int = 1
    action_information: Dict[bytes, ActionInformation] = field(default_factory=dict)
    @property
    def expected_return(self) -> float:
        return self.total_return / self.num_visits

class MCTS(metaclass=ABCMeta):
    def __init__(self, game: Game):
        self.game = game
        self.action_spec = game.game_spec.move_spec
        self.node_information: Dict[bytes, NodeInformation] = {}

    @abstractmethod
    def select_action(self, observation: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def select_random_action(self) -> np.ndarray:
        ...
    
    def rollout(self, game: Game):
        action = self.select_random_action()
        timestep = game.step(action)
        if timestep.step_type == StepType.LAST:
            return timestep.reward
        return (timestep.reward or 0) + self.rollout(game)

    def search(self, game: Optional[Game] = None):
        if not game:
            game = self.game
        observation = game.get_observation()
        state = observation.tobytes()
        if not state in self.node_information:
            # game will be modified in rollout
            returns = self.rollout(deepcopy(game))
            node_information = NodeInformation(
                game=game,
                num_visits=1,
                total_return=returns
            )
            self.node_information[state] = node_information
            return returns

        node_information = self.node_information[state]
        action = self.select_action(observation)
        action_representation = action.tobytes()
        if action_representation in node_information.action_information:
            action_information = node_information.action_information[action_representation]
            next_state = action_information.next_state
            # running the simulation is the expensive part
            returns = action_information.reward + (
                self.search(self.node_information[next_state].game)
                if not action_information.termination else 0
            )
        else:
            # only copied when stepping
            game = deepcopy(game)
            timestep = game.step(action)
            action_information = ActionInformation(
                next_state=timestep.observation.tobytes(),
                reward=timestep.reward,
                termination=timestep.step_type == StepType.LAST,
                num_visits=0
            )
            node_information.action_information[action_representation] = action_information
            returns = timestep.reward + (
                self.search(game) 
                if not action_information.termination else 0
            )
        node_information.num_visits += 1
        action_information.num_visits += 1
        node_information.total_return += returns
        return returns
from copy import deepcopy
import numpy as np

from src.utils import classproperty

from typing import List, Optional, Tuple, Union
from dm_env.specs import Array, BoundedArray
from abc import ABCMeta, abstractmethod
from dm_env import StepType, TimeStep
from dataclasses import dataclass

@dataclass
class GameSpec:
    move_spec: BoundedArray
    observation_spec: Union[Array, BoundedArray]
    def validate_action(self, action: np.ndarray) -> np.ndarray:
        assert action.shape == self.move_spec.shape
        if isinstance(self.move_spec, BoundedArray):
            assert (action >= self.move_spec.minimum).all()
            assert (action <= self.move_spec.maximum).all()
        return action.astype(self.move_spec.dtype)

    def validate_observation(self, observation: np.ndarray) -> np.ndarray:
        assert observation.shape == self.observation_spec.shape
        if isinstance(self.observation_spec, BoundedArray):
            assert (observation >= self.observation_spec.minimum).all()
            assert (observation <= self.observation_spec.maximum).all()
        return observation.astype(self.observation_spec.dtype)

class Game(metaclass=ABCMeta):
    game_spec: GameSpec = None
    to_play: int = None
    max_round = 0
    player_deltas: List[int] = [1, -1]
    eps: float = 1e-6 # used in the event of a "draw"

    @classproperty
    def num_players(cls):
        return len(cls.player_deltas)

    @abstractmethod
    def _get_observation(self)-> np.ndarray:
        ...

    def get_observation(self) -> np.ndarray:
        return self.validate_observation(self._get_observation())

    @abstractmethod
    def _step(self, action: np.ndarray, display: bool = False) -> float:
        ...

    def step(self, action: np.ndarray, display: bool = False) -> TimeStep:
        self.pre_step(action)
        reward = self._step(action, display=display)
        return self.post_step(reward)

    def pre_step(self, action: np.ndarray):
        action = self.validate_action(action)

    def post_step(self, reward: float) -> TimeStep:
        self.to_play = (self.to_play + 1) % self.num_players
        self.current_round += 1
        return TimeStep(
            step_type=self.get_step_type(),
            reward=reward,
            observation=self.get_observation(),
            discount=1,
        )

    @abstractmethod
    def _reset(self):
        ...

    def reset(self, starting_player: Optional[int]=None) -> TimeStep:
        self.pre_reset(starting_player)
        self._reset()
        return self.post_reset()

    def pre_reset(self, starting_player: Optional[int]=None):
        if starting_player is None:
            starting_player = np.random.randint(self.num_players)
        assert starting_player in np.arange(self.num_players)
        self.to_play = starting_player
        self.current_round = 0

    def post_reset(self) -> TimeStep:
        return TimeStep(
            step_type=StepType.FIRST,
            reward=None,
            observation=self.get_observation(),
            discount=1
        )

    def sample(self, action: np.ndarray) -> TimeStep:
        game = deepcopy(self)
        return game.step(action, display=False)

    def get_symmetries(self, player: int, observation: np.ndarray, action: np.ndarray, reward: float) -> List[Tuple[int, np.ndarray, np.ndarray, float]]:
        return Game.no_symmetries(player, observation, action, reward)

    @staticmethod
    def no_symmetries(player: int, observation: np.ndarray, action: np.ndarray, reward: float) -> List[Tuple[int, np.ndarray, np.ndarray, float]]:
        return [(player, observation, action, reward)]

    def validate_observation(self, observation: np.ndarray)-> np.ndarray:
        return self.game_spec.validate_observation(observation)

    def validate_action(self, action: np.ndarray)-> np.ndarray:
        return self.game_spec.validate_action(action)

    def get_step_type(self, round: Optional[int] = None) -> StepType:
        if round is None:
            round = self.current_round
        if self.current_round >= self.max_round:
            return StepType.LAST
        else:
            return StepType.MID

    @property
    def player_delta(self) -> int:
        return self.player_deltas[self.to_play]

    def get_random_move(self):
        return np.random.uniform(low=self.game_spec.move_spec.minimum, high=self.game_spec.move_spec.maximum)
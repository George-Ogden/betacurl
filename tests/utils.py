from src.io import SaveableObject, SaveableModel
from src.game import Game, GameSpec, Player

from .config import SAVE_DIR

from dm_env._environment import TimeStep
from dm_env.specs import BoundedArray
from typing import List, Optional, Tuple

import numpy as np
import os

class StubGame(Game):
    """game where each player's score is incremented by the minimum of their actions"""

    action_size = 3
    max_move = 10

    def __init__(self, rounds: int = 6):
        self.max_round = rounds
        self.game_spec = GameSpec(
            move_spec=BoundedArray(
                maximum=(self.max_move,) * self.action_size,
                minimum=(0,) * self.action_size,
                dtype=np.float32,
                shape=(self.action_size,),
            ),
            observation_spec=BoundedArray(
                minimum=(-self.max_round // 2 * self.max_move,),
                maximum=((self.max_round + 1) // 2 * self.max_move,),
                shape=(1,),
                dtype=np.float32,
            ),
        )
        self.reset()

    def _reset(self) -> TimeStep:
        self.score = [0, 0]

    def _get_observation(self)-> np.ndarray:
        return np.array(self.score[0] - self.score[1]).reshape(1)

    def _step(self, action: np.ndarray, display: bool = False) -> float:
        delta = np.min(action)
        self.score[self.to_play] += delta
        if self.to_play == 1:
            delta *= -1

        if display:
            print(*self.score)

        return delta

class SparseStubGame(StubGame):
    """stub game with rewadr only on last step"""
    def _step(self, action: np.ndarray, display: bool = False) -> Optional[float]:
        super()._step(action, display)
        reward = float(self.get_observation())
        if self.current_round == self.max_round - 1:
            return reward if reward != 0 else self.eps * -self.player_delta

class BadSymetryStubGame(StubGame):
    def get_symmetries(
        self, observation: np.ndarray, action: np.ndarray, reward: float
    ) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        return [(observation * 0 + 1, action * 0 + 1, 1), (observation * 0 - 1, action * 0 + 2, -1)]

class GoodPlayer(Player):
    def move(self, game: Game)-> np.ndarray:
        return game.game_spec.move_spec.maximum

class BadPlayer(Player):
    def move(self, game: Game)-> np.ndarray:
        return game.game_spec.move_spec.minimum

def generic_save_test(object: SaveableObject):
    object.save(SAVE_DIR)

    assert os.path.exists(SAVE_DIR)
    assert os.path.exists(os.path.join(SAVE_DIR, object.DEFAULT_FILENAME))
    assert os.path.getsize(os.path.join(SAVE_DIR, object.DEFAULT_FILENAME)) > 0
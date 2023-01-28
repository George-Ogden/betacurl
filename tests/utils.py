from src.game import Player
from src.game import Game, GameSpec

from dm_env._environment import TimeStep
from dm_env.specs import BoundedArray
from typing import List, Tuple

from pytest import fixture, mark
from glob import glob
import os

import numpy as np

SAVE_DIR = "test_save_dir"

requires_cleanup = mark.usefixtures("cleanup")
slow = mark.skip(reason="taking too long")

# def equal_defaultdicts(dict1, dict2):
#     if dict1.keys() != dict2.keys():
#         return False
#     for k in dict1:
#         if type(dict1[k]) != type(dict2[k]):
#             return False
#         if type(dict1[k]) == np.ndarray:
#             if dict1[k].shape != dict2[k].shape or (dict1[k] != dict2[k]).all():
#                 return False
#         else:
#             if dict1[k] != dict2[k]:
#                 return False
#     return True


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

@fixture
def cleanup():
    yield 
    cleanup_dir(SAVE_DIR)

def cleanup_dir(dir: str):
    for filename in glob(f"{dir}/*"):
        if os.path.isdir(filename):
            cleanup_dir(filename)
        else:
            os.remove(filename)
    
    if os.path.exists(dir):
        os.rmdir(dir)
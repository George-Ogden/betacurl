from tensorflow.keras import callbacks
from dm_env import TimeStep
from copy import deepcopy
import numpy as np
import os

from typing import List, Optional, Tuple
from dm_env.specs import BoundedArray

from src.utils import SaveableObject
from src.game import Game, GameSpec
from src.player import Player

from .config import SAVE_DIR

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
        self.score = [0., 0.]

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

    def get_symmetries(self, player: int, observation: np.ndarray, action: np.ndarray, reward: float) -> List[Tuple[int, np.ndarray, np.ndarray, float]]:
        return [(player, observation, action, reward), (-player, -observation, action, -reward)]

class SparseStubGame(StubGame):
    """stub game with reward only on last step"""
    def _step(self, action: np.ndarray, display: bool = False) -> Optional[float]:
        super()._step(action, display)
        reward = float(self.get_observation()[0])
        if self.current_round == self.max_round - 1:
            return reward if reward != 0 else self.eps * -self.player_delta

class BadSymetryStubGame(StubGame):
    def get_symmetries(
        self, player: int, observation: np.ndarray, action: np.ndarray, reward: float
    ) -> List[Tuple[int, np.ndarray, np.ndarray, float]]:
        return [(player, observation * 0 + 1, action * 0 + 1, 1), (player, observation * 0 - 1, action * 0 + 2, -1)]

class MDPStubGame(StubGame):
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
                minimum=(-self.max_round // 2 * self.max_move, 0),
                maximum=((self.max_round + 1) // 2 * self.max_move, self.max_round),
                shape=(2,),
                dtype=np.float32,
            ),
        )
        self.reset()

    def _get_observation(self)-> np.ndarray:
        return np.array((self.score[0] - self.score[1], self.current_round))

    def get_symmetries(self, player: int, observation: np.ndarray, action: np.ndarray, reward: float) -> List[Tuple[int, np.ndarray, np.ndarray, float]]:
        return Game.no_symmetries(player, observation, action, reward)

class MDPSparseStubGame(SparseStubGame):
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
                minimum=(-self.max_round // 2 * self.max_move, 0),
                maximum=((self.max_round + 1) // 2 * self.max_move, self.max_round),
                shape=(2,),
                dtype=np.float32,
            ),
        )
        self.reset()

    def _get_observation(self)-> np.ndarray:
        return np.array((self.score[0] - self.score[1], self.current_round))

class BinaryStubGame(MDPSparseStubGame):
    def _step(self, action: np.ndarray, display: bool = False) -> Optional[float]:
        """win: +1, loss: -1"""
        super()._step(action, display)
        reward = float(self.get_observation()[0])
        if self.current_round == self.max_round - 1:
            return np.sign(reward)

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

def save_load(object: SaveableObject) -> SaveableObject:
    object.save(SAVE_DIR)
    new_object = type(object).load(SAVE_DIR)
    assert type(new_object) == type(object)
    return new_object

def generic_save_load_test(object: SaveableObject, excluded_attrs: List[str] = []) -> Tuple[SaveableObject, SaveableObject]:
    original = deepcopy(object)
    copy = save_load(object)
    for attr in vars(original):
        if attr in excluded_attrs:
            continue
        original_attr = getattr(original, attr)
        copy_attr = getattr(copy, attr)
        error = f"original.{attr} != copy.{attr}, ({original_attr} != {copy_attr})"
        try:
            assert original_attr == copy_attr, error
        except ValueError:
            assert type(original_attr) == type(copy_attr), error
    return original, copy
    
class EpochCounter(callbacks.Callback):
    def on_train_begin(self, *args, **kwargs):
        self.counter = 0
    
    def on_epoch_begin(self, *args, **kwargs):
        self.counter += 1

def find_hidden_size(layers):
    for layer in layers:
        if hasattr(layer, "units"):
            if layer.units == 63:
                return True
        elif hasattr(layer, "layers"):
            if find_hidden_size(layer.layers):
                return True
    return False
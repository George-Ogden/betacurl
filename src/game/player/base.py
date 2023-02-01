from ...io import SaveableObject
from ..game import GameSpec, Game

from abc import ABCMeta, abstractmethod

import numpy as np

class Player(SaveableObject, metaclass=ABCMeta):
    DEFAULT_FILENAME = "player.pickle"
    def __init__(self, game_spec: GameSpec):
        self.game_spec = game_spec
        self.train()

    def train(self) -> "Self":
        self.is_training = True
        return self

    def eval(self) -> "Self":
        self.is_training = False
        return self

    def dummy_constructor(self, game_spec: GameSpec) -> "Self":
        return self

    @abstractmethod
    def move(self, game: Game)-> np.ndarray:
        ...
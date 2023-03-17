import numpy as np

from ..game import Game, GameSpec

from .base import Player

class RandomPlayer(Player):
    def __init__(self, game_spec: GameSpec):
        self.minimum = game_spec.move_spec.minimum
        self.maximum = game_spec.move_spec.maximum

    def move(self, game: Game)-> np.ndarray:
        return np.random.uniform(low=self.minimum, high=self.maximum)
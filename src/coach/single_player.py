from tqdm import trange
import numpy as np
import wandb
import os

from typing import List, Optional, Tuple, Type
from copy import copy

from ..player import Arena, Player, NNMCTSPlayer, NNMCTSPlayerConfig
from ..game import Game, GameSpec
from ..utils import SaveableObject

from .config import CoachConfig
from .coach import Coach

class SinglePlayerCoach(Coach):
    def __init__(
        self,
        game: Game,
        config: CoachConfig = CoachConfig()
    ):
        super().__init__(game=game, config=config)
        assert self.game.num_players == 1, f"the `{type(self).__name__}` class is for single player games only"

    def benchmark(self, Opponent: Type[Player]) -> Tuple[int, int]:
        arenas = [
            Arena([self.player.dummy_constructor], game=self.game.clone()),
            Arena([Opponent], game=self.game)
        ]
        results = [0, 0]
        for _ in trange(self.num_eval_games, desc="Playing games"):
            returns = [
                arena.play_game(
                    starting_player=0,
                    display=False,
                    return_history=False,
                    training=False
                ) for arena in arenas
            ]
            results[np.argmax(returns)] += 1
        return tuple(results)
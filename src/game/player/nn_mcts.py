from __future__ import annotations

from typing import Optional, Type

from copy import deepcopy
import numpy as np

from ...mcts import MCTSModel, NNMCTS, BEST_MCTS
from ...model import Learnable

from ..game import Game, GameSpec

from .config import MCTSPlayerConfig
from .mcts import MCTSPlayer

class NNMCTSPlayer(MCTSPlayer, Learnable):
    def __init__(
        self,
        game_spec: GameSpec,
        config: Optional[MCTSPlayerConfig]=MCTSPlayerConfig()
    ):
        super().__init__(
            game_spec,
            NNMCTS,
            config=config
        )
        self.trees = []
        self.model: Optional[MCTSModel] = None

    # def train(self) -> "Self":
    #     self.save_mcts()
    #     return super().train()

    # def eval(self) -> "Self":
    #     self.save_mcts()
    #     return super().eval()
    
    # def save_mcts(self):
    #     if self.mcts is not None:
    #         self.trees.append(self.mcts)

    def create_mcts(self, game: Game) -> NNMCTS:
        return self.MCTSClass(
            game=game,
            model=self.model,
            config=self.config.mcts_config
        )
    
    def learn(self, training_data):
        self.model.learn(training_data)
from __future__ import annotations

from typing import Optional, Type

from copy import deepcopy
import numpy as np

from ..mcts import MCTS, Node, BEST_MCTS
from ..game import Game, GameSpec

from .config import MCTSPlayerConfig
from .base import Player

class MCTSPlayer(Player):
    def __init__(
        self,
        game_spec: GameSpec,
        MCTSClass: Type[MCTS] = BEST_MCTS,
        config: Optional[MCTSPlayerConfig]=MCTSPlayerConfig()
    ):
        self.config = deepcopy(config)
        self.simulations = config.num_simulations
        self.temperature = 1.
        
        self.MCTSClass = MCTSClass
        self.mcts: Optional[MCTS] = None

        super().__init__(game_spec)

    def train(self) -> "Self":
        self.mcts = None
        self.temperature = 1.
        return super().train()

    def eval(self) -> "Self":
        self.mcts = None
        self.temperature = 0.
        return super().eval()
    
    def get_current_node(self, game: Game) -> Node:
        return self.mcts.get_node(game.get_observation())

    def create_mcts(self, game: Game) -> MCTS:
        return self.MCTSClass(
            game,
            config=self.MCTSClass.CONFIG_CLASS(
                **self.config.mcts_config
            )
        )

    def move(self, game: Game) -> np.ndarray:
        if self.mcts is None:
            self.mcts = self.create_mcts(game)
        else:
            self.mcts.cleanup(game)

        for _ in range(self.simulations):
            self.mcts.search(game)

        actions, probs = self.mcts.get_action_probs(game, temperature=self.temperature)
        return actions[np.random.choice(len(actions), p=probs)]
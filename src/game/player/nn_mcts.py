from typing import Optional

from ...mcts import MCTSModel, NNMCTS

from ..game import Game, GameSpec

from .config import NNMCTSPlayerConfig
from .mcts import MCTSPlayer

class NNMCTSPlayer(MCTSPlayer):
    def __init__(
        self,
        game_spec: GameSpec,
        config: Optional[NNMCTSPlayerConfig]=NNMCTSPlayerConfig()
    ):
        super().__init__(
            game_spec,
            NNMCTS,
            config=config
        )

        self.scaling_spec = config.scaling_spec
        self.model: Optional[MCTSModel] = None

    def create_mcts(self, game: Game) -> NNMCTS:
        return self.MCTSClass(
            game=game,
            model=self.model,
            config=self.config.mcts_config
        )
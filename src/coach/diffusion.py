from ..player import NNMCTSPlayer, NNMCTSPlayerConfig
from ..mcts import DiffusionMCTSModel
from ..game import GameSpec

from .coach import Coach

class DiffusionCoach(Coach):
    def create_player(
        self,
        game_spec: GameSpec,
        config: NNMCTSPlayerConfig = NNMCTSPlayerConfig()
    ) -> NNMCTSPlayer:
        return NNMCTSPlayer(
            game_spec=game_spec,
            config=config,
            ModelClass=DiffusionMCTSModel
        )
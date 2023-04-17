from typing import Optional, Type

from ..mcts import DiffusionMCTSModel
from ..game import Game

from .config import CoachConfig
from .coach import Coach

class DiffusionCoach(Coach):
    def __init__(
        self,
        game: Game,
        config: CoachConfig=CoachConfig(),
        ModelClass: Optional[Type[DiffusionMCTSModel]]=DiffusionMCTSModel,
    ):
        super().__init__(game, config=config, ModelClass=ModelClass)
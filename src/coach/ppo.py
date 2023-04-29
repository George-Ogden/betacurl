from typing import Type

from ..mcts import PPOMCTSModel
from ..game import Game

from .single_player import SinglePlayerCoach
from .config import SinglePlayerCoachConfig

class PPOCoach(SinglePlayerCoach):
    def __init__(
        self,
        game: Game,
        config: SinglePlayerCoachConfig = SinglePlayerCoachConfig(),
        ModelClass: Type[PPOMCTSModel] = PPOMCTSModel,
    ):
        super().__init__(game=game, config=config, ModelClass=ModelClass)
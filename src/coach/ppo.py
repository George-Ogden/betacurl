from typing import Type

from ..mcts import PPOMCTSModel
from ..game import Game

from .single_player import SinglePlayerCoach
from .config import PPOCoachConfig

class PPOCoach(SinglePlayerCoach):
    def __init__(
        self,
        game: Game,
        config: PPOCoachConfig = PPOCoachConfig(),
        ModelClass: Type[PPOMCTSModel] = PPOMCTSModel,
    ):
        super().__init__(game=game, config=config, ModelClass=ModelClass)
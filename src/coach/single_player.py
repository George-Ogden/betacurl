import numpy as np

from typing import Type

from ..player import Arena, Player
from ..game import Game
from ..mcts import ReinforceMCTSModel

from .config import CoachConfig
from .coach import Coach

class SinglePlayerCoach(Coach):
    def __init__(
        self,
        game: Game,
        config: CoachConfig=CoachConfig(),
        ModelClass: Type[ReinforceMCTSModel]=ReinforceMCTSModel,
    ):
        super().__init__(game=game, config=config, ModelClass=ModelClass)
        assert self.game.num_players == 1, f"the `{type(self).__name__}` class is for single player games only"

    def compare(self, Opponent: Type[Player]) -> bool:
        arena = Arena([self.player.dummy_constructor], game=self.game.clone())
        current_player_results = np.array(arena.play_games(self.num_eval_games, display=False, training=False))
        arena = Arena([Opponent], game=self.game.clone())
        opponent_results = np.array(arena.play_games(self.num_eval_games, display=False, training=False))
        wins = (current_player_results >= opponent_results).sum()
        return wins > self.win_threshold
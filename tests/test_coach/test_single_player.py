from copy import copy
import numpy as np

from pytest import mark
import os

from src.player import MCTSPlayer, NNMCTSPlayerConfig
from src.coach import CoachConfig, SinglePlayerCoach
from src.game import Game, MujocoGame
from src.mcts import MCTSConfig

from tests.utils import FixedValueMCTS
from tests.config import SAVE_DIR

necessary_config = {
    "save_directory": SAVE_DIR,

}

time_limit = MujocoGame.time_limit
MujocoGame.time_limit = 1000
game = MujocoGame("cartpole", "swingup")
MujocoGame.time_limit = time_limit

game_spec = game.game_spec
observation_spec = game_spec.observation_spec
move_spec = game_spec.move_spec

class GoodMCTS(FixedValueMCTS):
    def __init__(self, game: Game, config: MCTSConfig = MCTSConfig()):
        super().__init__(game, config, move=move_spec.maximum)

class BadMCTS(FixedValueMCTS):
    def __init__(self, game: Game, config: MCTSConfig = MCTSConfig()):
        super().__init__(game, config, move=np.zeros(move_spec.shape, move_spec.dtype))

class BadPlayerCoach(SinglePlayerCoach):
    @property
    def best_player(self):
        if os.path.exists(self.best_checkpoint_path):
            best_player = self.load_player(self.best_checkpoint_path)
        else:
            config = copy(self.config.player_config)
            best_player = MCTSPlayer(
                self.game.game_spec,
                MCTSClass=BadMCTS,
                config=config
            )

        self.current_best = best_player
        return best_player

class GoodPlayerCoach(SinglePlayerCoach):
    @property
    def best_player(self):
        if os.path.exists(self.best_checkpoint_path):
            best_player = self.load_player(self.best_checkpoint_path)
        else:
            config = copy(self.config.player_config)
            best_player = MCTSPlayer(
                self.game.game_spec,
                MCTSClass=GoodMCTS,
                config=config
            )

        self.current_best = best_player
        return best_player

@mark.probabilistic
@mark.slow
def test_benchmark():
    coach = BadPlayerCoach(
        game=game,
        config=CoachConfig(
            **necessary_config,
            evaluation_games=1,
            player_config=NNMCTSPlayerConfig(
                num_simulations=5
            )
        )
    )
    assert not coach.compare(
        MCTSPlayer(
            game_spec,
            MCTSClass=GoodMCTS,
            config=copy(coach.config.player_config),
        ).dummy_constructor
    )
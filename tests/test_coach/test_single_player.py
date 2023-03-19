from copy import copy
from glob import glob
import numpy as np
import os

from pytest import mark

from src.player import Arena, MCTSPlayer, NNMCTSPlayer, NNMCTSPlayerConfig
from src.coach import CoachConfig, SinglePlayerCoach
from src.mcts import MCTS, MCTSConfig
from src.model import TrainingConfig
from src.game import Game, MujocoGame

from tests.config import cleanup, requires_cleanup, SAVE_DIR

necessary_config = {
    "save_directory": SAVE_DIR,

}

game = MujocoGame("cartpole", "swingup")
game_spec = game.game_spec
observation_spec = game_spec.observation_spec
move_spec = game_spec.move_spec

class FixedValueMCTS(MCTS):
    def __init__(self, game: Game, config: MCTSConfig = MCTSConfig(), move = None):
        super().__init__(game, config)
        self.move = move

    def select_action(self, observation: np.ndarray) -> np.ndarray:
        return self.move.copy()

    def _get_action_probs(self, game: Game, temperature: float):
        return np.array([self.select_action(None)]), np.array([1.])

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
            evaluation_games=4
        )
    )
    wins, losses = coach.benchmark(
        MCTSPlayer(
            game_spec,
            MCTSClass=GoodMCTS,
            config=copy(coach.config.player_config)
        ).dummy_constructor
    )
    assert wins + losses == 4
    assert wins == 4

@mark.probabilistic
@mark.slow
@requires_cleanup
def test_model_learns():
    coach = SinglePlayerCoach(
        game=game,
        config=CoachConfig(
            resume_from_checkpoint=False,
            num_games_per_episode=10,
            num_iterations=2,
            training_config=TrainingConfig(
                lr=1e-3,
                training_epochs=5
            ),
            player_config=NNMCTSPlayerConfig(
                num_simulations=15
            )
        )
    )

    coach.learn()

    arena = Arena(
        game=game,
        players=[
            coach.best_player.dummy_constructor, NNMCTSPlayer(
                game_spec=game.game_spec,
                config=NNMCTSPlayerConfig(
                    num_simulations=3
                )
            ).dummy_constructor
        ]
    )
    wins, losses = arena.play_games(10)
    assert wins >= 7
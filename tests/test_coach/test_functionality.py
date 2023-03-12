from copy import copy
from glob import glob
import os

from pytest import mark

from src.game import Arena, Coach, CoachConfig, Coach, CoachConfig, NNMCTSPlayer, NNMCTSPlayerConfig, RandomPlayer
from src.model import  TrainingConfig

from tests.config import cleanup, requires_cleanup, SAVE_DIR
from tests.utils import BadPlayer, BinaryStubGame, GoodPlayer, MDPStubGame, MDPSparseStubGame

special_cases = dict(
    evaluation_games=4,
    win_threshold=.5,
    successive_win_requirement=4,
)

necessary_config = {
    "save_directory": SAVE_DIR,
}

config_dict = dict(
    resume_from_checkpoint=False,
    num_games_per_episode=2,
    num_iterations=2,
    train_buffer_length=1,
    **necessary_config,
    **special_cases,
    training_config=TrainingConfig(
        training_epochs=10,
        batch_size=64,
        lr=1e-1,
        training_patience=20
    )
)
custom_training_config = copy(config_dict["training_config"])
custom_training_config.training_epochs = 5

stub_game = MDPStubGame(6)
sparse_stub_game = MDPSparseStubGame(6)
observation_spec = stub_game.game_spec.observation_spec
move_spec = stub_game.game_spec.move_spec

boring_coach = Coach(
    game=stub_game,
    config=CoachConfig(
        **config_dict
    )
)

class BadPlayerCoach(Coach):
    @property
    def best_player(self):
        if os.path.exists(self.best_checkpoint_path):
            best_player = self.load_player(self.best_checkpoint_path)
        else:
            config = copy(self.config.player_config)
            config.scaling_spec = -stub_game.max_move
            best_player = BadPlayer(
                self.game.game_spec,
                config
            )

        self.current_best = best_player
        return best_player

class GoodPlayerCoach(Coach):
    @property
    def best_player(self):
        if os.path.exists(self.best_checkpoint_path):
            best_player = self.load_player(self.best_checkpoint_path)
        else:
            config = copy(self.config.player_config)
            config.scaling_spec = stub_game.max_move * 2
            best_player = GoodPlayer(
                self.game.game_spec,
                config
            )

        self.current_best = best_player
        return best_player

def test_reward_transformed_correctly():
    transform = set(boring_coach.transform_history_for_training(
        [(1, 0, 0, 0), (-1, 10, 10, 1), (1, 20, 20, 2), (-1, 30, 30, 3)],
    ))
    assert transform == set([(1, 0, 0, 6), (-1, 10, 10, 6), (1, 20, 20, 5), (-1, 30, 30, 3)])

def test_reward_transformed_correctly_with_None():
    transform = set(boring_coach.transform_history_for_training(
        [(1, 0, 0, None), (-1, 10, 10, None), (1, 20, 20, None), (-1, 30, 30, 3)],
    ))
    assert transform == set([(1, 0, 0, 3), (-1, 10, 10, 3), (1, 20, 20, 3), (-1, 30, 30, 3)])

@requires_cleanup
def test_no_default_best():
    coach = Coach(
        game=stub_game,
        config=CoachConfig(
            num_iterations=0,
            num_games_per_episode=2,
            **necessary_config
        )
    )
    coach.learn()
    assert not os.path.exists(coach.best_checkpoint_path)

@requires_cleanup
def test_sparse_game_for_coaching():
    coach = GoodPlayerCoach(
        game=sparse_stub_game,
        config=CoachConfig(
            num_iterations=1,
            train_buffer_length=1,
            num_games_per_episode=2,
            evaluation_games=10,
            win_threshold=.6,
            **necessary_config
        )
    )
    coach.learn()
    assert isinstance(coach.best_player, GoodPlayer)

    coach = BadPlayerCoach(
        game=sparse_stub_game,
        config=CoachConfig(
            num_iterations=1,
            train_buffer_length=1,
            num_games_per_episode=2,
            evaluation_games=10,
            win_threshold=.6,
            **necessary_config
        )
    )
    coach.learn()
    assert not isinstance(coach.best_player, BadPlayer)

@requires_cleanup
def test_learning_patience():
    coach = GoodPlayerCoach(
        game=sparse_stub_game,
        config=CoachConfig(
            num_iterations=10,
            successive_win_requirement=4,
            train_buffer_length=20,
            num_games_per_episode=2,
            evaluation_games=10,
            win_threshold=.6,
            **necessary_config
        )
    )
    coach.learn()
    assert isinstance(coach.best_player, GoodPlayer)
    assert len(glob(f"{SAVE_DIR}/*")) == 5

@requires_cleanup
def test_logs_format(capsys):
    coach = Coach(
        game=stub_game,
        config=CoachConfig(
            num_iterations=5,
            train_buffer_length=5,
            num_games_per_episode=2,
            evaluation_games=4,
            win_threshold=.6,
            **necessary_config
        )
    )
    coach.learn()

    output = capsys.readouterr()
    assert not "{" in output
    assert not "}" in output

@mark.probabilistic
@mark.slow
@requires_cleanup
def test_model_learns():
    max_move = BinaryStubGame.max_move
    BinaryStubGame.max_move = 1
    game = BinaryStubGame()
    coach = Coach(
        game=game,
        config=CoachConfig(
            resume_from_checkpoint=False,
            num_games_per_episode=100,
            num_iterations=2,
            training_config=TrainingConfig(
                lr=1e-3,
                training_epochs=2
            ),
            player_config=NNMCTSPlayerConfig(
                num_simulations=3
            ),
            use_intermediate_states=False
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
    wins, losses = arena.play_games(100)
    assert wins >= 60

    # cleanup
    BinaryStubGame.max_move = max_move
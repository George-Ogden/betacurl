import tensorflow as tf

from glob import glob
from copy import copy
import time

from pytest import mark
import os

from src.game import Coach, CoachConfig, Coach, CoachConfig, NNMCTSPlayer, NNMCTSPlayerConfig
from src.model import TrainingConfig

from tests.config import cleanup, cleanup_dir, requires_cleanup, SAVE_DIR
from tests.utils import MDPStubGame, MDPSparseStubGame

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

@requires_cleanup
def test_checkpoint_restored_correctly():
    coach = Coach(
        stub_game,
        config=CoachConfig(resume_from_checkpoint=True, **necessary_config)
    )
    coach.player.dummy_variable = 15
    coach.save_model(10, wins=0)

    new_coach = Coach(
        stub_game,
        config=CoachConfig(resume_from_checkpoint=True, **necessary_config)
    )
    iteration = new_coach.load_checkpoint()
    assert iteration == 10
    assert new_coach.player.dummy_variable == 15

@requires_cleanup
def test_checkpoint_restores_in_training():
    coach = Coach(
        stub_game,
        config=CoachConfig(
            resume_from_checkpoint=True,
            num_iterations=4,
            num_games_per_episode=2,
            **necessary_config
        )
    )
    coach.dummy_variable = 25
    coach.learn()

    del coach.dummy_variable
    update_time = time.time()
    cleanup_dir(coach.get_checkpoint_path(3))
    cleanup_dir(coach.get_checkpoint_path(4))

    coach.learn()

    assert os.path.getmtime(coach.get_checkpoint_path(0)) < update_time
    assert os.path.getmtime(coach.get_checkpoint_path(4)) > update_time
    assert coach.dummy_variable == 25

@requires_cleanup
def test_training_history_restored():
    coach = Coach(
        game=stub_game,
        config=CoachConfig(
            train_buffer_length=25,
            num_iterations=4,
            num_games_per_episode=2,
            **necessary_config
        )
    )
    coach.learn()
    del coach.train_example_history
    coach.load_checkpoint()
    assert len(coach.train_example_history) == 4

@requires_cleanup
def test_best_player_saves_and_loads():
    coach = Coach(
        game=stub_game,
        config=CoachConfig(
            train_buffer_length=25,
            num_iterations=1,
            evaluation_games=40,
            **necessary_config
        )
    )
    coach.learn()

    champion = NNMCTSPlayer(stub_game.game_spec)
    champion.dummy_variable = 22

    player = coach.player
    coach.player = champion
    coach.save_model(0, 40)
    coach.player = player

    best_player = coach.best_player
    assert best_player.dummy_variable == 22

@mark.slow
@requires_cleanup
def test_reloading_mcts_coach(capsys):
    coach = Coach(
        game=stub_game,
        config=CoachConfig(
            **necessary_config,
            num_games_per_episode=2,
            num_iterations=2,
            evaluation_games=4,
            player_config=NNMCTSPlayerConfig(
                num_simulations=4
            )
        )
    )
    coach.learn()
    capsys.readouterr()

    coach = Coach(
        game=stub_game,
        config=CoachConfig(
            **necessary_config,
            num_games_per_episode=1,
            num_iterations=3,
            resume_from_checkpoint=True,
            evaluation_games=4,
            player_config=NNMCTSPlayerConfig(
                num_simulations=4
            )
        )
    )
    coach.learn()
    
    assert coach.num_iterations == 3
    assert coach.num_games_per_episode == 1
    assert len(glob(f"{SAVE_DIR}/model-0*")) == 4, glob(f"{SAVE_DIR}/model-0*")
    
    output = capsys.readouterr()
    assert "starting iteration 2" in output.out.lower()
    assert not "starting iteration 1" in output.out.lower()

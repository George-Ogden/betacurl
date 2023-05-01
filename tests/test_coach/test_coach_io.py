from glob import glob
from copy import copy
import time
import os

from pytest import mark

from src.coach import Coach, CoachConfig, SinglePlayerCoach, SinglePlayerCoachConfig
from src.player import NNMCTSPlayerConfig
from src.model import TrainingConfig
from src.game import MujocoGame

from tests.config import cleanup, cleanup_dir, requires_cleanup, SAVE_DIR
from tests.utils import MDPStubGame, MDPSparseStubGame

necessary_config = dict(
    save_directory=SAVE_DIR,
)

config_dict = dict(
    num_games_per_episode=2,
    num_iterations=2,
    resume_from_checkpoint=False,
    player_config=NNMCTSPlayerConfig(
        num_simulations=2,
    ),
    warm_start_games=1,
    **necessary_config,
    training_config=TrainingConfig(
        training_epochs=10,
        batch_size=64,
        lr=1e-1,
        training_patience=20
    )
)
custom_training_config = copy(config_dict["training_config"])
custom_training_config.training_epochs = 5

single_config_dict = config_dict | dict(
    eval_games=2,
    eval_simulations=2,
)
del single_config_dict["warm_start_games"]

stub_game = MDPStubGame(6)
sparse_stub_game = MDPSparseStubGame(6)
observation_spec = stub_game.game_spec.observation_spec
move_spec = stub_game.game_spec.move_spec

time_limit = MujocoGame.time_limit
MujocoGame.time_limit = 1
single_player_game = MujocoGame("cartpole", "swingup")
MujocoGame.time_limit = time_limit

boring_coach = Coach(
    game=stub_game,
    config=CoachConfig(
        **config_dict
    )
)

@requires_cleanup
def test_checkpoint_restores():
    coach = Coach(
        stub_game,
        config=CoachConfig(resume_from_checkpoint=True, **necessary_config)
    )
    coach.player.dummy_variable = 15
    coach.save_model(10)

    new_coach = Coach(
        stub_game,
        config=CoachConfig(resume_from_checkpoint=True, **necessary_config)
    )
    iteration = new_coach.load_checkpoint()
    assert iteration == 10
    assert new_coach.player.dummy_variable == 15

@mark.slow
@requires_cleanup
def test_checkpoint_restores_in_training():
    coach = Coach(
        stub_game,
        config=CoachConfig(
            **(
                config_dict | dict(
                    resume_from_checkpoint=True,
                )
            )
        )
    )
    coach.dummy_variable = 25
    coach.learn()

    del coach.dummy_variable
    update_time = time.time()
    cleanup_dir(coach.get_checkpoint_path(1))
    cleanup_dir(coach.get_checkpoint_path(2))

    coach.learn()

    assert os.path.getmtime(coach.get_checkpoint_path(0)) < update_time
    assert os.path.getmtime(coach.get_checkpoint_path(2)) > update_time
    assert coach.dummy_variable == 25

@mark.slow
@requires_cleanup
def test_reloading_coach(capsys):
    config = copy(config_dict)
    config["num_iterations"] = 2
    config["resume_from_checkpoint"] = True
    coach = Coach(
        game=stub_game,
        config=CoachConfig(
            **config
        )
    )
    coach.learn()
    capsys.readouterr()

    config["num_games_per_episode"] = 1
    config["num_iterations"] = 3
    coach = Coach(
        game=stub_game,
        config=CoachConfig(
            **config
        )
    )
    coach.learn()
    
    assert coach.num_iterations == 3
    assert coach.num_games_per_episode == 1
    assert len(glob(f"{SAVE_DIR}/model-0*")) == 4, glob(f"{SAVE_DIR}/model-0*")
    
    captured = capsys.readouterr()
    assert "starting iteration 3" in captured.out.lower()
    assert not "starting iteration 2" in captured.out.lower()

@requires_cleanup
def test_save_frequency():
    coach = Coach(
        stub_game,
        config=CoachConfig(
            **(
                config_dict | dict(
                    save_frequency=2,
                )
            )
        )
    )
    coach.learn()
    assert len(glob(f"{SAVE_DIR}/model-0*")) == 2
    assert os.path.exists(f"{SAVE_DIR}/model-000000")
    assert os.path.exists(f"{SAVE_DIR}/model-000002")

@mark.slow
@requires_cleanup
def test_single_player_best_checkpoint():
    coach = SinglePlayerCoach(
        single_player_game,
        config=SinglePlayerCoachConfig(
            **(
                single_config_dict | dict(
                    best_checkpoint_path="best"
                )
            )
        )
    )
    coach.learn()

    assert os.path.exists(f"{SAVE_DIR}/best")
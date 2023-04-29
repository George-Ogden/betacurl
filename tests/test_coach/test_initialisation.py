from copy import copy
import numpy as np

from pytest import mark
import os

from src.coach import Coach, CoachConfig, PPOCoach, SinglePlayerCoach, SinglePlayerCoachConfig
from src.mcts import NNMCTSConfig, PPOMCTSModel
from src.player import NNMCTSPlayerConfig
from src.model import TrainingConfig
from src.game import MujocoGame

from tests.config import cleanup, requires_cleanup, SAVE_DIR
from tests.utils import MDPStubGame, MDPSparseStubGame

special_cases = dict(
    eval_games=4,
    eval_simulations=5,
)

necessary_config = {
    "save_directory": SAVE_DIR,
}

config_dict = dict(
    resume_from_checkpoint=False,
    num_games_per_episode=2,
    num_iterations=2,
    warm_start_games=1,
    **necessary_config,
    training_config=TrainingConfig(
        training_epochs=10,
        batch_size=64,
        lr=1e-1,
        training_patience=20
    ),
    player_config=NNMCTSPlayerConfig(
        num_simulations=2,
    )
)
single_config_dict = config_dict | special_cases
del single_config_dict["warm_start_games"]
custom_training_config = copy(config_dict["training_config"])
custom_training_config.training_epochs = 2

stub_game = MDPStubGame(6)
sparse_stub_game = MDPSparseStubGame(6)
observation_spec = stub_game.game_spec.observation_spec
move_spec = stub_game.game_spec.move_spec

time_limit = MujocoGame
MujocoGame.time_limit = 1
single_player_game = MujocoGame("point_mass", "easy")
swingup = MujocoGame("cartpole", "swingup")
MujocoGame.time_limit = time_limit

boring_coach = Coach(
    game=stub_game,
    config=CoachConfig(
        **config_dict
    )
)

def test_coach_saves_config():
    assert not os.path.exists(SAVE_DIR)
    for k, v in config_dict.items():
        assert getattr(boring_coach, k) == v

    assert type(boring_coach.game) == MDPStubGame

@mark.slow
@requires_cleanup
def test_coach_uses_training_config():
    coach = Coach(
        game=MDPStubGame(),
        config=CoachConfig(
            **config_dict
        )
    )

    modified_coach = Coach(
        game=MDPStubGame(),
        config=CoachConfig(
            **config_dict |
            {
                "training_config": custom_training_config
            }
        )
    )

    coach.learn()
    modified_coach.learn()

    model = coach.player.model.model
    modified_model = modified_coach.player.model.model
    assert model.history.params["epochs"] == 10
    assert modified_model.history.params["epochs"] == 2
    assert np.allclose(model.optimizer._learning_rate.numpy(), .1)

@requires_cleanup
def test_ppo_coach_uses_ppo_model():
    coach = PPOCoach(
        game=single_player_game,
        config=SinglePlayerCoachConfig(
            **(
                single_config_dict | dict(
                    player_config=NNMCTSPlayerConfig(
                        mcts_config=NNMCTSConfig()
                    )
                )
            )
        )
    )

    assert coach.player.ModelClass == PPOMCTSModel

@requires_cleanup
def test_coach_initial_model_states():
    coach = Coach(
        game=stub_game,
        config=CoachConfig(
            **config_dict |
            {"num_iterations":0}
        )
    )
    coach.learn()
    assert coach.player.model is not None

@requires_cleanup
def test_single_player_coach_initial_model_states():
    coach = SinglePlayerCoach(
        game=single_player_game,
        config=SinglePlayerCoachConfig(
            **single_config_dict |
            {"num_iterations":0}
        )
    )
    coach.learn()
    assert coach.player.model is not None

@requires_cleanup
def test_ppo_coach_initial_model_states():
    config = config_dict.copy()
    config["num_iterations"] = 0
    coach = PPOCoach(
        game=single_player_game,
        config=SinglePlayerCoachConfig(
            **single_config_dict
        )
    )
    coach.learn()
    assert coach.player.model is not None
    assert isinstance(coach.player.model, PPOMCTSModel)

@requires_cleanup
def test_coach_propagates_model():
    coach = SinglePlayerCoach(
        game=swingup,
        ModelClass=PPOMCTSModel,
        config=SinglePlayerCoachConfig(
            **necessary_config,
            player_config=NNMCTSPlayerConfig(
                mcts_config=NNMCTSConfig()
            )
        )
    )
    assert coach.player.ModelClass == PPOMCTSModel
from copy import copy
import numpy as np
import os

from src.coach import Coach, CoachConfig, DiffusionCoach, PPOCoach, PPOCoachConfig, SinglePlayerCoach
from src.mcts import NNMCTSConfig, PPOMCTSModel
from src.player import NNMCTSPlayerConfig
from src.model import TrainingConfig
from src.game import MujocoGame

from tests.config import cleanup, requires_cleanup, SAVE_DIR
from tests.utils import MDPStubGame, MDPSparseStubGame

special_cases = dict(
    evaluation_games=4,
    win_threshold=.5,
    successive_win_requirement=4,
    num_eval_simulations=5,
)

necessary_config = {
    "save_directory": SAVE_DIR,
}

config_dict = dict(
    resume_from_checkpoint=False,
    num_games_per_episode=2,
    num_iterations=2,
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

single_player_game = MujocoGame("point_mass", "easy")

boring_coach = Coach(
    game=stub_game,
    config=CoachConfig(
        **config_dict
    )
)

def test_coach_saves_config():
    assert not os.path.exists(SAVE_DIR)
    for k, v in config_dict.items():
        if k in special_cases:
            continue
        assert getattr(boring_coach, k) == v

    # special cases
    assert boring_coach.win_threshold == .5
    assert boring_coach.num_eval_games == 4
    assert boring_coach.learning_patience == 4
    assert boring_coach.eval_simulations == 5

    assert type(boring_coach.game) == MDPStubGame

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
    assert modified_model.history.params["epochs"] == 5
    assert np.allclose(model.optimizer._learning_rate.numpy(), .1)

@requires_cleanup
def test_ppo_coach_uses_ppo_model():
    copy_config = config_dict.copy()
    del copy_config["win_threshold"]
    coach = PPOCoach(
        game=single_player_game,
        config=PPOCoachConfig(
            **copy_config,
            player_config=NNMCTSPlayerConfig(
                mcts_config=NNMCTSConfig()
            )
        )
    )

    assert coach.player.ModelClass == PPOMCTSModel
    assert coach.best_player.ModelClass == PPOMCTSModel

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
    assert coach.best_player.model is None

@requires_cleanup
def test_single_player_coach_initial_model_states():
    coach = SinglePlayerCoach(
        game=single_player_game,
        config=CoachConfig(
            **config_dict |
            {"num_iterations":0}
        )
    )
    coach.learn()
    assert coach.best_player.model is None

@requires_cleanup
def test_ppo_coach_initial_model_states():
    coach = PPOCoach(
        game=single_player_game,
        config=CoachConfig(
            **config_dict |
            {"num_iterations":0}
        )
    )
    coach.learn()
    assert coach.best_player.model is not None
    assert isinstance(coach.best_player.model, PPOMCTSModel)

@requires_cleanup
def test_diffusion_coach_initial_model_states():
    coach = DiffusionCoach(
        game=stub_game,
        config=CoachConfig(
            **config_dict |
            {"num_iterations":0}
        )
    )
    coach.learn()
    assert coach.best_player.model is None

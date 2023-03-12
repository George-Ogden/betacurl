from copy import copy
import numpy as np
import os

from src.game import Arena, Coach, CoachConfig, Coach, CoachConfig
from src.model import TrainingConfig

from tests.utils import BinaryStubGame, MDPStubGame, MDPSparseStubGame
from tests.config import cleanup, requires_cleanup, SAVE_DIR

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

def test_coach_saves_config():
    assert not os.path.exists(SAVE_DIR)
    for k, v in config_dict.items():
        if k in special_cases:
            continue
        assert getattr(boring_coach, k) == v

    # special cases
    assert boring_coach.win_threshold == 2
    assert boring_coach.num_eval_games == 4
    assert boring_coach.learning_patience == 4

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
def test_coach_avoids_intermediate_states():
    game = BinaryStubGame()
    coach = Coach(
        game=game,
        config=CoachConfig(
            use_intermediate_states=False,
            **necessary_config
        )
    )

    arena = Arena([coach.player.dummy_constructor] * 2, coach.game)
    result, history = arena.play_game(display=False, return_history=True)
    transformed_history = coach.transform_history_for_training(history)
    observations = [bytes(observation) for player, observation, *data in history]
    for player, observation, action, reward in transformed_history:
        assert bytes(observation) in observations
        assert reward == result

@requires_cleanup
def test_coach_includes_intermediate_states():
    game = BinaryStubGame()
    coach = Coach(
        game=game,
        config=CoachConfig(
            use_intermediate_states=True,
            **necessary_config
        )
    )

    arena = Arena([coach.best_player.dummy_constructor] * 2, coach.game)
    result, history = arena.play_game(display=False, return_history=True)
    transformed_history = coach.transform_history_for_training(history)
    matches = non_matches = 0
    for *data, reward in transformed_history:
        if reward == result:
            matches += 1
        else:
            non_matches += 1
    assert non_matches > 0
    assert non_matches > matches / 2
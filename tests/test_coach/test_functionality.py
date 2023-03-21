from copy import copy
from glob import glob
import numpy as np
import os

from pytest import mark

from src.player import Arena, MCTSPlayer, NNMCTSPlayer, NNMCTSPlayerConfig
from src.coach import Coach, CoachConfig, PPOCoach, PPOCoachConfig
from src.mcts import MCTSConfig, NNMCTS, NNMCTSConfig
from src.model import TrainingConfig
from src.game import Game, MujocoGame

from tests.utils import BinaryStubGame, MDPStubGame, MDPSparseStubGame
from tests.config import cleanup, requires_cleanup, SAVE_DIR

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

boring_coach = Coach(
    game=stub_game,
    config=CoachConfig(
        **config_dict
    )
)

class FixedValueMCTS(NNMCTS):
    CONFIG_CLASS = NNMCTSConfig
    def __init__(self, game: Game, config: MCTSConfig = MCTSConfig(), move = None):
        super().__init__(game, config=config)
        self.move = move

    def select_action(self, observation: np.ndarray) -> np.ndarray:
        super().select_action(observation)
        return self.move.copy()

    def _get_action_probs(self, game: Game, temperature: float):
        return np.array([self.select_action(game.get_observation())]), np.array([1.])

class GoodMCTS(FixedValueMCTS):
    def __init__(self, game: Game, config: MCTSConfig = MCTSConfig()):
        super().__init__(game, config=config, move=game.game_spec.move_spec.maximum)

class BadMCTS(FixedValueMCTS):
    def __init__(self, game: Game, config: MCTSConfig = MCTSConfig()):
        super().__init__(game, config=config, move=game.game_spec.move_spec.minimum)

class BadPlayerCoach(Coach):
    @property
    def best_player(self):
        if os.path.exists(self.best_checkpoint_path):
            best_player = self.load_player(self.best_checkpoint_path)
        else:
            config = copy(self.config.player_config)
            config.scaling_spec = -stub_game.max_move
            best_player = MCTSPlayer(
                self.game.game_spec,
                MCTSClass=BadMCTS,
                config=config
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
            best_player = MCTSPlayer(
                self.game.game_spec,
                MCTSClass=GoodMCTS,
                config=config
            )

        self.current_best = best_player
        return best_player

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
            num_games_per_episode=2,
            evaluation_games=10,
            win_threshold=.6,
            **necessary_config
        )
    )
    coach.learn()
    assert coach.best_player.MCTSClass == GoodMCTS

    coach = BadPlayerCoach(
        game=sparse_stub_game,
        config=CoachConfig(
            num_iterations=1,
            num_games_per_episode=2,
            evaluation_games=10,
            win_threshold=.6,
            **necessary_config
        )
    )
    coach.learn()
    assert coach.best_player.MCTSClass != BadMCTS

@requires_cleanup
def test_transform():
    coach = Coach(
        game=copy(sparse_stub_game),
        config=CoachConfig(
            num_iterations=1,
            num_games_per_episode=2,
            evaluation_games=10,
            win_threshold=.6,
            **necessary_config
        )
    )
    game = coach.game
    game.discount = .9
    coach.current_best = coach.best_player
    arena = Arena([coach.current_best.dummy_constructor] * 2, game=game)
    result, game_history = arena.play_game(0, return_history=True)
    game.reset(0)
    history = coach.transform_history_for_training(game_history)
    previous_value = None
    p = None
    for player, observation, action, value, *data in history[::-1]:
        assert p is None or p == player * -1
        p = player
        assert np.allclose(observation, game.get_observation())
        game.step(action)
        if previous_value is not None:
            assert previous_value == value
        previous_value = value

@requires_cleanup
def test_train_examples_cleared_after_win():
    coach = GoodPlayerCoach(
        game=sparse_stub_game,
        config=CoachConfig(
            num_iterations=1,
            num_games_per_episode=2,
            evaluation_games=10,
            win_threshold=.6,
            **necessary_config
        )
    )
    coach.learn()
    assert coach.best_player.MCTSClass == GoodMCTS
    assert len(coach.train_example_history) > 0

    coach = BadPlayerCoach(
        game=sparse_stub_game,
        config=CoachConfig(
            num_iterations=1,
            num_games_per_episode=2,
            evaluation_games=10,
            win_threshold=.6,
            **necessary_config
        )
    )
    coach.learn()
    assert len(coach.train_example_history) == 0

@requires_cleanup
def test_learning_patience():
    coach = GoodPlayerCoach(
        game=sparse_stub_game,
        config=CoachConfig(
            num_iterations=10,
            successive_win_requirement=4,
            num_games_per_episode=2,
            evaluation_games=10,
            win_threshold=.6,
            **necessary_config
        )
    )
    coach.learn()
    assert coach.best_player.MCTSClass == GoodMCTS
    assert len(glob(f"{SAVE_DIR}/*")) == 5

@requires_cleanup
def test_eval_simulations_change():
    coach = Coach(
        game=sparse_stub_game,
        config=CoachConfig(
            num_iterations=1,
            num_games_per_episode=2,
            evaluation_games=4,
            win_threshold=.0,
            num_eval_simulations=5,
            player_config=NNMCTSPlayerConfig(
                num_simulations=20
            ),
            **necessary_config
        )
    )

    coach.learn()
    coach.update()

    assert coach.current_best.simulations == 5
    assert coach.best_player.simulations == 20
    assert coach.player.simulations == 20

@requires_cleanup
def test_logs_format(capsys):
    coach = Coach(
        game=stub_game,
        config=CoachConfig(
            num_iterations=5,
            num_games_per_episode=2,
            evaluation_games=4,
            win_threshold=.6,
            **necessary_config
        )
    )
    coach.learn()

    captured = capsys.readouterr()
    assert not "{" in captured.out
    assert not "{" in captured.err
    assert not "}" in captured.out
    assert not "}" in captured.err

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
            num_eval_simulations=3,
            **necessary_config,
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

@mark.probabilistic
@mark.slow
@requires_cleanup
def test_ppo_model_learns():
    time_limit = MujocoGame.time_limit
    discount = MujocoGame.discount
    MujocoGame.time_limit = 5
    MujocoGame.discount = .5
    game = MujocoGame(
        domain_name="point_mass",
        task_name="easy"
    )
    MujocoGame.time_limit = time_limit
    MujocoGame.discount = discount
    coach = PPOCoach(
        game=game,
        config=CoachConfig(
            resume_from_checkpoint=False,
            num_games_per_episode=3,
            num_iterations=3,
            training_config=TrainingConfig(
                lr=1e-2,
                training_epochs=5
            ),
            player_config=NNMCTSPlayerConfig(
                num_simulations=10,
                mcts_config=NNMCTSConfig(
                    cpw=.5,
                    kappa=1.,
                )
            ),
            num_eval_simulations=10,
            evaluation_games=1,
            **necessary_config,
        )
    )

    coach.learn()
    eval_game = MujocoGame(
        domain_name="point_mass",
        task_name="easy"
    )
    
    arena = Arena(
        game=eval_game,
        players=[
            coach.best_player.dummy_constructor, 
        ]
    )
    score = 0
    # needs running multiple times due to very sparse environment
    for _ in range(5):
        score = max(score, arena.play_game(return_history=False, training=False))
    assert score > 1.
from copy import copy, deepcopy
from dm_env import StepType
from glob import glob
import numpy as np
import os

from pytest import mark

from src.coach import Coach, CoachConfig, PPOCoach, PPOCoachConfig
from src.player import Arena, MCTSPlayer, NNMCTSPlayerConfig
from src.game import Game, MujocoGame
from src.model import TrainingConfig
from src.mcts import MCTSConfig

from tests.utils import MDPStubGame, MDPSparseStubGame, FixedValueMCTS
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
            best_player = self.player.load(self.best_checkpoint_path)
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
            best_player = self.player.load(self.best_checkpoint_path)
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

class EvalEnvSavingCoach(PPOCoach):
    def __init__(self, game: Game, config: PPOCoachConfig = PPOCoachConfig()):
        super().__init__(game, config)
        self.eval_envs = []
    def compare(self, Opponent) -> bool:
        self.eval_envs.append(deepcopy(self.eval_environment))
        return super().compare(Opponent)

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
    for player, observation, action, value, policy in history[::-1]:
        assert p is None or p == player * -1
        p = player
        assert np.allclose(observation, game.get_observation())
        game.step(action)
        if previous_value is not None:
            assert previous_value == value
        previous_value = value
        advantage_sum = 0.
        for action, advantage in policy:
            advantage_sum += advantage
        assert np.allclose(advantage_sum, 0.)

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

@mark.slow
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

@requires_cleanup
def test_eval_arena_is_constant():
    eval_envs = []
    time_limit = MujocoGame.time_limit
    timestep = MujocoGame.timestep
    MujocoGame.time_limit = 6.
    MujocoGame.timestep = .5
    game = MujocoGame(domain_name="point_mass", task_name="easy")
    MujocoGame.timestep = timestep
    MujocoGame.time_limit = time_limit

    coach = EvalEnvSavingCoach(
        game=game,
        config=PPOCoachConfig(
            num_iterations=3,
            num_games_per_episode=1,
            evaluation_games=1,
            num_eval_simulations=2,
            player_config=NNMCTSPlayerConfig(
                num_simulations=2
            ),
            **necessary_config
        )
    )

    coach.learn()
    
    move_spec = coach.game.game_spec.move_spec
    actions = [np.random.uniform(low=move_spec.minimum, high=move_spec.maximum) for _ in range(10)]
    timesteps = []
    for env in eval_envs:
        for i, action in enumerate(actions):
            timestep = env.step(action)
            if i == len(timesteps):
                timesteps.append(timestep)
            else:
                assert timestep.discount == timesteps[i].discount
                assert timestep.reward == timesteps[i].reward
                assert (timestep.observation == timesteps[i].observation).all()
                assert timestep.step_type == timesteps[i].step_type
            if timestep.step_type == StepType.LAST:
                env.reset()
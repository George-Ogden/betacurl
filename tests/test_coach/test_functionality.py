from copy import copy, deepcopy
import numpy as np

from src.coach import Coach, CoachConfig, PPOCoach, SinglePlayerCoachConfig
from src.player import Arena, NNMCTSPlayerConfig
from src.game import Game, MujocoGame
from src.model import TrainingConfig

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
    initial_lr=1e-1,
    final_lr=1e-3,
    initial_temperature=1.,
    final_temperature=0.25,
    training_config=TrainingConfig(
        training_epochs=10,
        batch_size=64,
        training_patience=20
    ),
    player_config=NNMCTSPlayerConfig(
        num_simulations=2,
    )
)
single_config_dict = config_dict | special_cases
del single_config_dict["warm_start_games"]

stub_game = MDPStubGame(6)
sparse_stub_game = MDPSparseStubGame(6)
move_spec = stub_game.game_spec.move_spec

time_limit = MujocoGame.time_limit
timestep = MujocoGame.timestep
MujocoGame.time_limit = 6.
MujocoGame.timestep = .5
mujoco_game = MujocoGame(domain_name="point_mass", task_name="easy")
MujocoGame.timestep = timestep
MujocoGame.time_limit = time_limit

boring_coach = Coach(
    game=stub_game,
    config=CoachConfig(
        **config_dict
    )
)

class EvalEnvSavingCoach(PPOCoach):
    def __init__(self, game: Game, config: SinglePlayerCoachConfig = SinglePlayerCoachConfig()):
        super().__init__(game, config)
        self.eval_envs = []
    def compare(self, Opponent) -> bool:
        self.eval_envs.append(deepcopy(self.eval_environment))
        return super().compare(Opponent)

@requires_cleanup
def test_transform():
    coach = Coach(
        game=copy(sparse_stub_game),
        config=CoachConfig(
            num_iterations=1,
            num_games_per_episode=2,
            **necessary_config
        )
    )
    game = coach.game
    game.discount = .9
    arena = Arena([coach.player.dummy_constructor] * 2, game=game)
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
        for action, advantage, num_visits in policy:
            advantage_sum += advantage
        assert np.allclose(advantage_sum, 0., atol=1e-5)

@requires_cleanup
def test_ppo_transform():
    coach = PPOCoach(
        game=copy(mujoco_game),
        config=SinglePlayerCoachConfig(
            **single_config_dict
        )
    )
    game = coach.game
    game.discount = .9
    arena = Arena([coach.player.dummy_constructor] * 2, game=game)
    result, game_history = arena.play_game(0, return_history=True)
    history = coach.transform_history_for_training(game_history)

    for *_, policy in history[::-1]:
        advantage_sum = 0.
        for action, advantage, num_visits in policy:
            advantage_sum += advantage
        assert np.allclose(advantage_sum, 0., atol=1e-5)

@requires_cleanup
def test_logs_format(capsys):
    coach = Coach(
        game=stub_game,
        config=CoachConfig(
            **config_dict
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
    game = mujoco_game

    coach = EvalEnvSavingCoach(
        game=game,
        config=SinglePlayerCoachConfig(
            **single_config_dict
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
            if timestep.step_type.last():
                env.reset()

@requires_cleanup
def test_temperature_decreases():
    boring_coach.learn()
    temp = boring_coach.player.temperature
    assert np.abs(
        temp - config_dict["initial_temperature"]
    ) > np.abs(
        temp - config_dict["final_temperature"]
    )
    boring_coach.player.train()
    temp = boring_coach.player.temperature
    assert np.abs(
        temp - config_dict["initial_temperature"]
    ) > np.abs(
        temp - config_dict["final_temperature"]
    )

@requires_cleanup
def test_lr_decreases():
    boring_coach.learn()
    lr = boring_coach.player.model.model.optimizer._learning_rate.numpy()
    assert np.abs(
        lr - config_dict["initial_lr"]
    ) > np.abs(
        lr - config_dict["final_lr"]
    )
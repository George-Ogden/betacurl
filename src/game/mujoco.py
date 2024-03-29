from dm_control.rl.control import Environment, flatten_observation, FLAT_OBSERVATION_KEY
from dm_env.specs import BoundedArray
from dm_control import suite
from copy import copy
import numpy as np

from .game import Game, GameSpec

class MujocoGame(Game):
    player_deltas = [1]
    discount = .95
    time_limit = 10.
    timestep = .1
    def __init__(self, domain_name: str, task_name: str):
        super().__init__()
        self.name = (domain_name, task_name)
        self.env: Environment = suite.load(
            domain_name=domain_name,
            task_name=task_name,
            task_kwargs={
                "random": True
            },
            environment_kwargs={
                "control_timestep": self.timestep,
                "flat_observation": True
            }
        )
        self.game_spec = GameSpec(
            move_spec=self.env.action_spec(),
            observation_spec=self.env.observation_spec()[FLAT_OBSERVATION_KEY],
            value_spec=BoundedArray(
                minimum=0,
                maximum=1 / (1 - self.discount),
                shape=(),
                dtype=np.float32
            )
        )
        self.max_round = int(self.time_limit / self.timestep)
        self.reset()
    
    def _reset(self):
        self.env.reset()
    
    def _get_observation(self) -> np.ndarray:
        observation = self.env.task.get_observation(self.env.physics)
        flat_observation = flatten_observation(observation)[FLAT_OBSERVATION_KEY]
        return flat_observation

    def _step(self, action: np.ndarray, display: bool = False) -> float:
        timestep = self.env.step(action)
        if display:
            print(timestep)
        return timestep.reward or 0

    def clone(self) -> "Self":
        # reduce memory by only copying the parts that change
        clone = copy(self)
        clone.env = copy(self.env)
        clone.env._physics = self.env._physics.copy(share_model=True)
        return clone
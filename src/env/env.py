from dm_control import suite
from copy import deepcopy
import numpy as np
from typing import Any
from collections import OrderedDict

from dm_control.rl.control import Environment
from dm_env._environment import TimeStep
from numpy.typing import ArrayLike

class EnvironmentWithSampling(Environment):
    def __init__(self, env: Environment):
        env.reset()
        for attr in vars(env):
            setattr(self, attr, getattr(env, attr))

    def sample(self, action: ArrayLike) -> TimeStep:
        # Create a copy of the environment
        env_copy = deepcopy(self)
        # Apply the action to the copied environment
        time_step = env_copy.step(action)
        # Return the new state
        return time_step
    
    def set_state(self, physics_state: np.ndarray):
        self.physics.set_state(physics_state)
    
    @staticmethod
    def load_mujoco(domain_name: str, task_name: str, ):
        mujoco_env = suite.load(domain_name, task_name, environment_kwargs={"flat_observation": True})
        return EnvironmentWithSampling(mujoco_env)

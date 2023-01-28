from src.env import EnvironmentWithSampling
from tests.utils import equal_defaultdicts

import numpy as np

def env_action_test(env, action):
    env = EnvironmentWithSampling(env)
    sampled_time_step = env.sample(action)
    time_step = env.step(action)
    assert equal_defaultdicts(sampled_time_step.observation, time_step.observation)
    assert time_step.reward == sampled_time_step.reward
    assert sampled_time_step.reward is not None

def test_mujoco_loading():
    env = EnvironmentWithSampling.load_mujoco("cartpole", "swingup")
    assert env._task._swing_up == True

def test_single_action():
    env = EnvironmentWithSampling.load_mujoco("cartpole", "swingup")
    action = 1
    env_action_test(env, action)

def test_single_multiple_actions():
    env = EnvironmentWithSampling.load_mujoco("point_mass", "easy")
    action = np.array([1, -1])
    env_action_test(env, action)

def test_set_state():
    environment = EnvironmentWithSampling.load_mujoco("cartpole", "swingup")
    env = EnvironmentWithSampling(environment)
    env.set_state(np.zeros(4))
    assert (env.physics.position() == 0).all()
    assert (env.physics.velocity() == 0).all()
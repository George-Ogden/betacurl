from src.evaluation import EvaluationStrategy
from src.sampling import SamplingStrategy
from src.env import EnvironmentWithSampling
from src.agent.config import AgentConfig
from dm_control import viewer

from copy import copy
import numpy as np

from typing import Callable, Iterable, Optional
from dm_env.specs import BoundedArray
from dm_env._environment import TimeStep
from collections import OrderedDict
from numpy.typing import ArrayLike

class Agent:
    def __init__(self, env: EnvironmentWithSampling, SamplingStrategyClass: Callable[[BoundedArray, BoundedArray], SamplingStrategy], EvaluationStrategyClass: Callable[[BoundedArray], EvaluationStrategy], config: Optional[AgentConfig]=AgentConfig()):
        self.config = copy(config)
        self.epsilon = config.epsilon
        self.gamma = config.gamma
        self.num_samples = config.num_samples

        self.env = env
        self.sampler: SamplingStrategy = SamplingStrategyClass(action_spec=env.action_spec(), observation_spec=env.observation_spec()["observations"])
        self.evaluator: EvaluationStrategy = EvaluationStrategyClass(observation_spec=env.observation_spec()["observations"])
    
    def evaluate_action(self, action: ArrayLike) -> float:
        time_step = self.env.sample(action)
        reward = time_step.reward
        value = self.evaluator.evaluate(time_step.observation["observations"])
        return reward + value * self.gamma

    def get_best_action_from_samples(self, observation: np.ndarray, potential_actions: Iterable[ArrayLike]) -> np.ndarray:
        self.env.set_state(observation)
        next_time_steps = [self.env.sample(action) for action in potential_actions]
        next_observations = np.stack([time_step.observation["observations"] for time_step in next_time_steps], axis=0)
        rewards = np.array([time_step.reward for time_step in next_time_steps])
        predicted_rewards = self.evaluator.evaluate(next_observations)
        rewards += self.gamma * predicted_rewards
        best_action = potential_actions[rewards.argmax()]
        return best_action
    
    def get_action(self, observation: OrderedDict, best: bool = False) -> np.ndarray:
        observation = observation["observations"]
        potential_actions = self.sampler.generate_actions(observation, n=self.num_samples)
        if not best and np.random.random() < self.epsilon:
            return potential_actions[np.random.randint(len(potential_actions))]
        return self.get_best_action_from_samples(observation, potential_actions)
    
    def on_policy(self, time_step: TimeStep):
        return self.get_action(time_step.observation, best=True)
    
    def preview(self):
        viewer.launch(self.env, self.on_policy)
    
    def evaluate(self) -> float:
        time_step = self.env.reset()
        reward = 0
        while not time_step.last():
            action = self.on_policy(time_step)
            time_step = self.env.step(action)
            reward += time_step.reward
        return reward
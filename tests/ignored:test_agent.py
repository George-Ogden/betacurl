from src.agent import Agent
from src.env import EnvironmentWithSampling
from src.agent import AgentConfig
from src.evaluation import EvaluationStrategy
from src.sampling import RandomSamplingStrategy, SamplingStrategy

from dm_control import suite
import numpy as np

swingup = suite.load("cartpole", "swingup", environment_kwargs={"flat_observation": True}, task_kwargs={"time_limit": 1})
sparse_swingup = suite.load("cartpole", "swingup_sparse", environment_kwargs={"flat_observation": True}, task_kwargs={"time_limit": 1})
setable_env = suite.load("point_mass", "easy", environment_kwargs={"flat_observation": True}, task_kwargs={"time_limit": 1})


class StubEvaluationStrategy(EvaluationStrategy):
    def evaluate(self, observation, **kwargs) -> float:
        return 1

stub_env = EnvironmentWithSampling(env=swingup)
stub_sparse_env = EnvironmentWithSampling(env=sparse_swingup)
setable_stub_env = EnvironmentWithSampling(env=setable_env)

config = AgentConfig(num_samples=10, epsilon=1, gamma=1)

sparse_agent = Agent(env=stub_sparse_env, SamplingStrategyClass=RandomSamplingStrategy, EvaluationStrategyClass=StubEvaluationStrategy, config=config)
agent = Agent(env=stub_env, SamplingStrategyClass=RandomSamplingStrategy, EvaluationStrategyClass=StubEvaluationStrategy, config=config)
setable_agent = Agent(env=setable_stub_env, SamplingStrategyClass=RandomSamplingStrategy, EvaluationStrategyClass=StubEvaluationStrategy, config=config)
zero_agent = Agent(env=setable_stub_env, SamplingStrategyClass=SamplingStrategy, EvaluationStrategyClass=StubEvaluationStrategy, config=config)

observation = {"observations": np.array([0,1,0,0])}

def test_evaluate_no_reward():
    evaluation = sparse_agent.evaluate_action(-1)
    assert evaluation == 1

def test_evaluate_with_reward():
    evaluation = agent.evaluate_action(-1)
    assert evaluation > 1

def test_get_best_action_from_samples():
    action = setable_agent.get_best_action_from_samples(observation["observations"], [0, 2])
    assert action in [0, 2]

def test_get_best_action():
    action = zero_agent.get_action(observation, True)
    assert (action == 0).all()

def test_get_action():
    action = zero_agent.get_action(observation, False)
    assert (action == 0).all()

def test_on_policy():
    action = setable_agent.on_policy(setable_agent.env.reset())
    assert action.shape == setable_agent.sampler.action_shape
    assert (setable_agent.sampler.action_range[0] <= action).all()
    assert (action <= setable_agent.sampler.action_range[1]).all()

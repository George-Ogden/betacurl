from __future__ import annotations

from typing import Any, Callable, Iterable, List, Optional, Tuple, Union
from dm_env.specs import BoundedArray
from numpy.typing import ArrayLike

from copy import copy
import numpy as np

from ...sampling import NNSamplingStrategy, SamplerConfig, SamplingStrategy
from ...evaluation import EvaluationStrategy, NNEvaluationStrategy
from ...model import Learnable, TrainingConfig

from ..game import Game, GameSpec

from .config import SamplingEvaluatingPlayerConfig
from .base import Player

class SamplingEvaluatingPlayer(Player, Learnable):
    SEPARATE_ATTRIBUTES = ["evaluator", "sampler"]
    def __init__(
        self,
        game_spec: GameSpec,
        SamplingStrategyClass: Callable[[BoundedArray, BoundedArray], SamplingStrategy] = NNSamplingStrategy,
        EvaluationStrategyClass: Callable[[BoundedArray], EvaluationStrategy] = NNEvaluationStrategy,
        config: Optional[SamplingEvaluatingPlayerConfig]=SamplingEvaluatingPlayerConfig()
    ):
        self.config = copy(config)
        self.epsilon = config.epsilon
        self.num_train_samples = config.num_train_samples
        self.num_eval_samples = config.num_eval_samples

        super().__init__(game_spec)

        self.setup_sampler_evaluator(
            game_spec=game_spec,
            SamplingStrategyClass=SamplingStrategyClass,
            EvaluationStrategyClass=EvaluationStrategyClass,
            config=config.sampler_config
        )

    def setup_sampler_evaluator(
        self,
        game_spec: GameSpec,
        SamplingStrategyClass: Callable[[BoundedArray, BoundedArray], SamplingStrategy] = NNSamplingStrategy,
        EvaluationStrategyClass: Callable[[BoundedArray], EvaluationStrategy] = NNEvaluationStrategy,
        config = Union[SamplerConfig, dict]
    ):
        sampler_config = SamplingStrategyClass.CONFIG_CLASS(**config)
            
        self.sampler: SamplingStrategy = SamplingStrategyClass(action_spec=game_spec.move_spec, observation_spec=game_spec.observation_spec, config=sampler_config)
        self.evaluator: EvaluationStrategy = EvaluationStrategyClass(observation_spec=game_spec.observation_spec)

    def train(self) -> "Self":
        self.num_samples = self.num_train_samples
        return super().train()

    def eval(self) -> "Self":
        self.num_samples = self.num_eval_samples
        return super().eval()

    def evaluate(self, observations: np.ndarray) -> Union[float, np.ndarray]:
        return self.evaluator.evaluate(observations)
    
    def generate_actions(self, observation: np.ndarray, n: Optional[int] = None) -> np.ndarray:
        return self.sampler.generate_actions(observation, n)

    def move(self, game: Game) -> np.ndarray:
        potential_actions = self.generate_actions(observation=game.get_observation(), n=self.num_samples)
        if self.is_training and np.random.random() < self.epsilon:
            return potential_actions[np.random.randint(len(potential_actions))]
        return self.get_best_move_from_samples(game, potential_actions)

    def get_best_move_from_samples(self, game: Game, potential_actions: Iterable[ArrayLike]) -> np.ndarray:
        next_time_steps = [game.sample(action) for action in potential_actions]
        next_observations = np.stack([time_step.observation for time_step in next_time_steps], axis=0)
        rewards = np.array([time_step.reward for time_step in next_time_steps], dtype=np.float32)
        if np.isnan(rewards).any():
            rewards[np.isnan(rewards)] = self.evaluate(next_observations[np.isnan(rewards)])
        best_action = potential_actions[(rewards * game.player_delta).argmax()]
        return best_action

    def learn(
        self,
        training_history: List[Tuple[int, np.ndarray, np.ndarray, float]],
        augmentation_function: Callable[[np.ndarray, np.ndarray, float], List[Tuple[np.ndarray, np.ndarray, float]]],
        training_config: TrainingConfig = TrainingConfig()
    ):
        if isinstance(self.evaluator, Learnable):
            self.evaluator.learn(training_history, augmentation_function, training_config)
        if isinstance(self.sampler, Learnable):
            self.sampler.learn(training_history, augmentation_function, training_config)
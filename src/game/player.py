from __future__ import annotations

from src.evaluation import EvaluationStrategy, NNEvaluationStrategy
from src.sampling import SamplingStrategy, NNSamplingStrategy
from src.game.game import GameSpec, Game
from src.io import SaveableObject, TrainingConfig

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from abc import ABCMeta, abstractmethod
from dm_env.specs import BoundedArray
from numpy.typing import ArrayLike
from dataclasses import dataclass

from copy import copy
import numpy as np

class Player(SaveableObject, metaclass=ABCMeta):
    DEFAULT_FILENAME = "player.pickle"
    def __init__(self, game_spec: GameSpec):
        self.game_spec = game_spec
        self.train()

    def train(self) -> "Self":
        self.is_training = True
        return self

    def eval(self) -> "Self":
        self.is_training = False
        return self

    def dummy_constructor(self, game_spec: GameSpec) -> "Self":
        return self

    @abstractmethod
    def move(self, game: Game)-> np.ndarray:
        ...

class RandomPlayer(Player):
    def __init__(self, game_spec: GameSpec):
        self.minimum = game_spec.move_spec.minimum
        self.maximum = game_spec.move_spec.maximum

    def move(self, game: Game)-> np.ndarray:
        return np.random.uniform(low=self.minimum, high=self.maximum)

@dataclass
class SamplingEvaluatingPlayerConfig:
    num_train_samples: int = 100
    """number of samples generated during training"""
    num_eval_samples: int = 100
    """number of samples generated during evaluation"""
    epsilon: float = 0.1
    """epsilon-greedy exploration parameter"""
    latent_size: Optional[int] = None
    """size of latent space for sample generation"""

class SamplingEvaluatingPlayer(Player):
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

        latent_variable = {} if config.latent_size is None else dict(latent_size=config.latent_size)
        self.sampler: SamplingStrategy = SamplingStrategyClass(action_spec=game_spec.move_spec, observation_spec=game_spec.observation_spec, **latent_variable)
        self.evaluator: EvaluationStrategy = EvaluationStrategyClass(observation_spec=game_spec.observation_spec)

        self.sampler_type = type(self.sampler)
        self.evaluator_type = type(self.evaluator)

    def train(self) -> SamplingEvaluatingPlayer:
        self.num_samples = self.num_train_samples
        return super().train()

    def eval(self) -> SamplingEvaluatingPlayer:
        self.num_samples = self.num_eval_samples
        return super().eval()

    def move(self, game: Game) -> np.ndarray:
        potential_actions = self.sampler.generate_actions(observation=game.get_observation(), n=self.num_samples)
        if self.is_training and np.random.random() < self.epsilon:
            return potential_actions[np.random.randint(len(potential_actions))]
        return self.get_best_move_from_samples(game, potential_actions)

    def get_best_move_from_samples(self, game: Game, potential_actions: Iterable[ArrayLike]) -> np.ndarray:
        next_time_steps = [game.sample(action) for action in potential_actions]
        next_observations = np.stack([time_step.observation for time_step in next_time_steps], axis=0)
        rewards = np.array([time_step.reward for time_step in next_time_steps], dtype=np.float32)
        if np.isnan(rewards).any():
            rewards[np.isnan(rewards)] = self.evaluator.evaluate(next_observations[np.isnan(rewards)])
        best_action = potential_actions[(rewards * game.player_delta).argmax()]
        return best_action

    def save(self, directory: str):
        sampler = self.sampler
        evaluator = self.evaluator
        self.sampler = None
        self.evaluator = None

        super().save(directory)

        self.sampler = sampler
        self.evaluator = evaluator

        evaluator.save(directory)
        sampler.save(directory)

    @classmethod
    def load(cls, directory: str) -> SamplingEvaluatingPlayer:
        player = super().load(directory)
        player.evaluator = player.evaluator_type.load(directory)
        player.sampler = player.sampler_type.load(directory)
        return player

    def learn(
        self,
        training_history: List[Tuple[int, np.ndarray, np.ndarray, float]],
        augmentation_function: Callable[[np.ndarray, np.ndarray, float], List[Tuple[np.ndarray, np.ndarray, float]]],
        training_config: TrainingConfig = TrainingConfig()
    ):
        if hasattr(self.evaluator, "learn"):
            self.evaluator.learn(training_history, augmentation_function, training_config)
        if hasattr(self.sampler, "learn"):
            self.sampler.learn(training_history, augmentation_function, training_config)
from __future__ import annotations

from typing import Any, Callable, Iterable, List, Optional, Tuple, Union
from dm_env.specs import BoundedArray
from numpy.typing import ArrayLike

from copy import copy
import numpy as np

from ...sampling import SamplerConfig, SharedTorsoSamplingStrategy
from ...model import TrainingConfig

from ..game import GameSpec

from .sampling_evaluating import SamplingEvaluatingPlayer
from .config import SamplingEvaluatingPlayerConfig
from .base import Player

class SharedTorsoSamplingEvaluatingPlayer(SamplingEvaluatingPlayer):
    SEPARATE_ATTRIBUTES = ["sampler_evaluator"]
    def __init__(
        self,
        game_spec: GameSpec,
        config: Optional[SamplingEvaluatingPlayerConfig]=SamplingEvaluatingPlayerConfig()
    ):
        super().__init__(
            game_spec=game_spec,
            config=config.sampler_config
            )
    
    def setup_sampler_evaluator(
        self,
        game_spec: GameSpec,
        SamplingStrategyClass = None,
        EvaluationStrategyClass = None,
        config = Union[SamplerConfig, dict]
    ):
        self.sampler_evaluator = SharedTorsoSamplingStrategy(
            action_spec=game_spec.move_spec,
            observation_spec=game_spec.observation_spec,
            config=config
        )

    def learn(
        self,
        training_history: List[Tuple[int, np.ndarray, np.ndarray, float]],
        augmentation_function: Callable[[np.ndarray, np.ndarray, float], List[Tuple[np.ndarray, np.ndarray, float]]],
        training_config: TrainingConfig = TrainingConfig()
    ):
        self.sampler_evaluator.learn(training_history, augmentation_function, training_config)
from tensorflow_probability import distributions
from tensorflow.keras import callbacks
from tensorflow import data
import tensorflow as tf
import numpy as np

from dm_env.specs import BoundedArray
from copy import copy

from typing import Callable, List, Optional, Tuple, Union
from abc import ABCMeta, abstractmethod

from ...model import CustomDecorator, TrainingConfig
from ...utils import SaveableMultiModel
from ...game import GameSpec

from .config import MCTSModelConfig

class MCTSModel(SaveableMultiModel, CustomDecorator, metaclass=ABCMeta):
    def __init__(
        self,
        game_spec: GameSpec,
        config: MCTSModelConfig = MCTSModelConfig()
    ):
        action_spec = game_spec.move_spec
        observation_spec = game_spec.observation_spec

        self.action_range = np.stack((action_spec.minimum, action_spec.maximum), axis=0, dtype=np.float32)
        self.action_shape = action_spec.shape
        self.observation_range = (
            np.stack((observation_spec.minimum, observation_spec.maximum), axis=0)
            if isinstance(observation_spec, BoundedArray)
            else None
        )
        self.observation_shape = observation_spec.shape

        self.config = copy(config)
        self.feature_size = config.feature_size
        self.max_grad_norm = config.max_grad_norm
        self.vf_coeff = config.vf_coeff

    @abstractmethod
    def predict_values(
        self,
        observation: Union[tf.Tensor, np.ndarray],
        training: bool=False
    ) -> Union[tf.Tensor, np.ndarray]:
        ...

    @abstractmethod
    def generate_distribution(
        self,
        observation: Union[tf.Tensor, np.ndarray],
        training: bool=False
    ) -> distributions.Distribution:
        ...

    def learn(
        self,
        training_data: List[Tuple[int, np.ndarray, np.ndarray, float, List[Tuple[np.ndarray, float]]]],
        augmentation_function: Callable[[int, np.ndarray, np.ndarray, float], List[Tuple[int, np.ndarray, np.ndarray, float]]],
        training_config: TrainingConfig = TrainingConfig()
    ) -> callbacks.History:
        dataset = self.preprocess_data(
            training_data,
            augmentation_function,
            training_config
        )

        return self.fit(dataset, training_config)

    def preprocess_data(
        self,
        training_data: List[Tuple[int, np.ndarray, np.ndarray, float, List[Tuple[np.ndarray, float]]]],
        augmentation_function: Callable[[int, np.ndarray, np.ndarray, float], List[Tuple[int, np.ndarray, np.ndarray, float]]],
        training_config: TrainingConfig = TrainingConfig()
    ) -> data.Dataset:
        return self.create_dataset(training_data)
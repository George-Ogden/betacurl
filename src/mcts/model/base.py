from tensorflow_probability import distributions
from tensorflow.keras import callbacks
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
        scaling_spec: Optional[np.ndarray] = None,
        config: MCTSModelConfig = MCTSModelConfig()
    ):
        action_spec = game_spec.move_spec
        observation_spec = game_spec.observation_spec

        self.action_range = np.stack((action_spec.minimum, action_spec.maximum), axis=0)
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

        if scaling_spec is None:
            self.scaling_spec = np.stack(
                (self.action_range.mean(axis=0), np.zeros(self.action_shape)),
                axis=-1
            )
        elif scaling_spec.ndim == 1:
            self.scaling_spec = np.stack(
                (scaling_spec, np.zeros(self.action_shape)),
                axis=-1
            )
        else:
            self.scaling_spec = scaling_spec.copy()
            self.scaling_spec[:, 1] = np.log(scaling_spec[:, 1])

        assert self.scaling_spec.shape == self.action_shape + (2,)

    @abstractmethod
    def predict_values(
        self,
        observation: Union[tf.Tensor, np.ndarray],
        training: bool=False
    ) -> Union[tf.Tensor, np.ndarray]:
        ...

    def generate_distribution(
        self,
        observation: Union[tf.Tensor, np.ndarray],
        training: bool=False
    ) -> distributions.Distribution:

        batch_throughput = True
        if observation.ndim == len(self.observation_shape):
            batch_throughput = False
            observation = np.expand_dims(observation, 0)

        features = self.feature_extractor(observation, training=training)
        raw_actions = self.policy_head(features, training=training)

        if not batch_throughput:
            raw_actions = tf.squeeze(raw_actions, 0)

        return self._generate_distribution(raw_actions)

    @abstractmethod
    def _generate_distribution(self, raw_actions: tf.Tensor) -> distributions.Distribution:
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
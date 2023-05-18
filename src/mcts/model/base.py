from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability import distributions
from tensorflow.keras import callbacks
from tensorflow import data
import tensorflow as tf
import numpy as np

from dm_env.specs import BoundedArray
from copy import copy

from typing import Callable, List, Tuple, Union
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

        self.value_coefficients = tf.constant(
            np.array([-1, 0, 1], dtype=np.float32)
            if game_spec.value_spec is None
            else np.arange(
                np.floor(self.scale_values(game_spec.value_spec.minimum)),
                np.ceil(self.scale_values(game_spec.value_spec.maximum)) + 1,
                dtype=np.result_type(game_spec.value_spec.dtype, np.float32)
            )
        )

        self.config = copy(config)
        self.feature_size = config.feature_size
        self.max_grad_norm = config.max_grad_norm
        self.vf_coeff = config.vf_coeff
        self.model = None

    def __init_subclass__(cls, **kwargs):
        # allow post_init method: https://stackoverflow.com/a/72593763/12103577
        def init_decorator(previous_init):
            def new_init(self, *args, **kwargs):
                previous_init(self, *args, **kwargs)
                if type(self) == cls:
                    self.__post_init__()
            return new_init

        cls.__init__ = init_decorator(cls.__init__)

    def __post_init__(self):
        self.setup_model()
    
    def setup_model(self):
        ...
    
    def save(self, path: str):
        model = self.model
        self.model = None
        super().save(path)
        self.model = model

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
        assert len(dataset) > 0, "Dataset is empty"

        return self.fit(dataset, training_config)

    def preprocess_data(
        self,
        training_data: List[Tuple[int, np.ndarray, np.ndarray, float, List[Tuple[np.ndarray, float]]]],
        augmentation_function: Callable[[int, np.ndarray, np.ndarray, float], List[Tuple[int, np.ndarray, np.ndarray, float]]],
        training_config: TrainingConfig = TrainingConfig()
    ) -> data.Dataset:
        return self.create_dataset(training_data)
    
    @staticmethod
    def inverse_scale_values(values: tf.Tensor) -> tf.Tensor:
        return tf.sign(values) * (
            (
                (
                    tf.sqrt(
                        1 + 4 * 0.001 * (tf.abs(values) + 1 + 0.001)
                    ) - 1
                ) / (2 * 0.001)
            ) ** 2 - 1
        )
    
    @staticmethod
    def scale_values(values: tf.Tensor) -> tf.Tensor:
        return tf.sign(values) * (
            tf.sqrt(
                tf.abs(values) + 1
            ) - 1
        ) + 0.001 * values

    def logits_to_values(self, logits: tf.Tensor) -> tf.Tensor:
        """convert logits to values

        Args:
            logits (tf.Tensor): support logits

        Returns:
            tf.Tensor: scaled values
        """
        return tf.reduce_sum(
            logits * self.value_coefficients,
            axis=-1
        )

    def values_to_logits(self, values: tf.Tensor) -> tf.Tensor:
        """convert values to logits

        Args:
            values (tf.Tensor): scaled values

        Returns:
            tf.Tensor: distribution of support coefficients
        """
        upper_bounds = tf.searchsorted(
            self.value_coefficients,
            values,
            side="left"
        )
        lower_bounds = upper_bounds - 1
        # linear interpolate between lower and upper bound values
        interpolation = (
            values - tf.gather(self.value_coefficients, lower_bounds)
        ) / (
            tf.gather(self.value_coefficients, upper_bounds) - tf.gather(self.value_coefficients, lower_bounds)
        )
        interpolation = interpolation[:, tf.newaxis]
        logits = tf.one_hot(
            lower_bounds,
            depth=len(self.value_coefficients),
            dtype=tf.float32
        ) * (1 - interpolation) + tf.one_hot(
            upper_bounds,
            depth=len(self.value_coefficients),
            dtype=tf.float32
        ) * interpolation
        return logits
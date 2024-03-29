from tensorflow_probability import distributions
from tensorflow.keras import callbacks
from tensorflow import data
import tensorflow as tf
import numpy as np

from dm_env.specs import BoundedArray
from copy import copy

from typing import Callable, Optional, List, Tuple, Type, Union
from abc import ABCMeta, abstractmethod

from ...utils import support_to_value, value_to_support
from ...model import CustomDecorator, TrainingConfig
from ...distribution import DistributionFactory
from ...utils import SaveableMultiModel
from ...game import GameSpec

from .config import MCTSModelConfig

class MCTSModel(SaveableMultiModel, CustomDecorator, metaclass=ABCMeta):
    CONFIG_CLASS = MCTSModelConfig
    def __init__(
        self,
        game_spec: GameSpec,
        DistributionFactory: Type[DistributionFactory],
        config: Optional[MCTSModelConfig] = None,
    ):
        if config is None:
            config = self.CONFIG_CLASS()

        self.distribution_factory = DistributionFactory(
            move_spec=game_spec.move_spec,
            config=DistributionFactory.CONFIG_CLASS(
                **(
                    config.distribution_config
                    or {}
                )
            )
        )

        action_spec = game_spec.move_spec
        observation_spec = game_spec.observation_spec

        self.action_range = np.stack((action_spec.minimum, action_spec.maximum), axis=0, dtype=np.float32)
        self.action_shape = action_spec.shape
        self.action_dim = self.action_range.ndim
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
                dtype=np.float32
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
        self.distribution_factory.noise_off()
        dataset = self.preprocess_data(
            training_data,
            augmentation_function,
            training_config
        )
        assert len(dataset) > 0, "Dataset is empty"

        history = self.fit(dataset, training_config)
        self.distribution_factory.noise_on()
        return history

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
        values = tf.cast(values, dtype=tf.float32)
        return tf.sign(values) * (
            tf.sqrt(
                tf.abs(values) + 1
            ) - 1
        ) + 0.001 * values

    def logits_to_values(self, logits: tf.Tensor) -> tf.Tensor:
        return support_to_value(logits, self.value_coefficients)

    def values_to_logits(self, values: tf.Tensor) -> tf.Tensor:
        return value_to_support(values, self.value_coefficients)
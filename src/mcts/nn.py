from tensorflow_probability import distributions
from tensorflow.keras import layers
from tensorflow import keras
from dm_env import StepType
import tensorflow as tf
import numpy as np

from typing import Dict, Optional, Tuple, Union

from ..model import DenseModelFactory, ModelFactory, BEST_MODEL_FACTORY
from ..io import SaveableMultiModel

from .config import MCTSModelConfig, NNMCTSConfig
from .fixed import FixedMCTS

class MCTSModel(SaveableMultiModel):
    models = {
        "feature_extractor": "feature_extractor.h5",
        "policy_head": "policy.h5",
        "value_head": "value.h5",
        "observation_head": "observation.h5",
    }
    def __init__(
        self,
        game_spec: "GameSpec",
        scaling_spec: Optional[np.ndarray] = None,
        model_factory: ModelFactory = BEST_MODEL_FACTORY,
        config: MCTSModelConfig = MCTSModelConfig()
    ):
        action_spec = game_spec.move_spec
        observation_spec = game_spec.observation_spec

        self.action_range = np.stack((action_spec.minimum, action_spec.maximum), axis=0)
        self.action_shape = action_spec.shape
        self.observation_range = np.stack((observation_spec.minimum, observation_spec.maximum), axis=0)
        self.observation_shape = observation_spec.shape

        self.feature_size = config.feature_size
        
        self.feature_extractor = model_factory.create_model(
            input_shape=self.observation_shape,
            output_shape=self.feature_size,
            config=model_factory.CONFIG_CLASS(
                output_activation="linear"
            )
        )
        self.feature_extractor = keras.Sequential(
            [
                layers.Rescaling(
                    scale=1./np.diff(self.observation_range, axis=0).squeeze(0),
                    offset=-self.observation_range.mean(axis=0),
                ),
                self.feature_extractor
            ]
        )

        self.policy_head = DenseModelFactory.create_model(
            input_shape=self.feature_size,
            output_shape=self.action_shape + (2,)
        )

        self.value_head = DenseModelFactory.create_model(
            input_shape=self.feature_size,
            output_shape=()
        )

        inputs = layers.Concatenate()(
            [   
                keras.Input(self.feature_size),
                layers.Reshape(
                    (np.prod(self.action_shape, dtype=int),)
                )(
                    keras.Input(self.action_shape)
                )
            ]
        )

        self.observation_head = DenseModelFactory.create_model(
            input_shape=self.feature_size + np.prod(action_spec.shape),
            output_shape=self.feature_size
        )(inputs)

        if scaling_spec is None:
            scaling_spec = np.stack(
                (self.action_range.mean(axis=0), np.zeros(self.action_shape)),
                axis=-1
            )
        elif scaling_spec.ndim == 1:
            scaling_spec = np.stack(
                (scaling_spec, np.zeros(self.action_shape)),
                axis=-1
            )
        else:
            scaling_spec = scaling_spec.copy()
            scaling_spec[:, 1] = np.log(scaling_spec[:, 1])

        assert scaling_spec.shape == self.action_shape + (2,)

        self.policy_head = keras.Sequential(
            [self.policy_head, layers.Rescaling(offset=scaling_spec, scale=1.)]
        )
    
    def predict_values(self, observation: Union[tf.Tensor, np.ndarray]) -> Union[tf.Tensor, np.ndarray]:
        batch_throughput = True
        if observation.ndim == len(self.observation_shape):
            batch_throughput = False
            observation = np.expand_dims(observation, 0)

        features = self.feature_extractor(observation)
        values = self.value_head(features)

        if not batch_throughput:
            values = tf.squeeze(values, 0).numpy()
        
        return values

    def generate_distribution(self, observation: Union[tf.Tensor, np.ndarray]) -> distributions.Distribution:
        batch_throughput = True
        if observation.ndim == len(self.observation_shape):
            batch_throughput = False
            observation = np.expand_dims(observation, 0)

        features = self.feature_extractor(observation)
        raw_actions = self.policy_head(features)

        if not batch_throughput:
            raw_actions = tf.squeeze(raw_actions, 0).numpy()

        means, log_stds = tf.split(raw_actions, 2, axis=-1)
        means = tf.squeeze(means, -1)
        log_stds = tf.squeeze(log_stds, -1)
        stds = tf.exp(log_stds)

        return distributions.TruncatedNormal(means, stds, *self.action_range)
from tensorflow.keras import callbacks
import tensorflow as tf
import numpy as np

from typing import Callable, List, Optional, Tuple
from dm_env.specs import BoundedArray

from ...model import ModelDecorator, ModelFactory, TrainingConfig, BEST_MODEL_FACTORY

from ..config import NNSamplerConfig
from ..base import SamplingStrategy

class NNSamplingStrategy(SamplingStrategy, ModelDecorator):
    CONFIG_CLASS = NNSamplerConfig
    DEFAULT_MODEL_FILE = "sampler.h5"
    def __init__(self, action_spec: BoundedArray, observation_spec: BoundedArray, model_factory: ModelFactory = BEST_MODEL_FACTORY, config: NNSamplerConfig = NNSamplerConfig()):
        super().__init__(action_spec, observation_spec, config)
        
        self.latent_size = config.latent_size
        self.setup_model(action_spec, observation_spec, model_factory, self.latent_size)

    def setup_model(self, action_spec: BoundedArray, observation_spec: BoundedArray, model_factory: ModelFactory, latent_size: int = 0) -> tf.keras.Model:
        config = BEST_MODEL_FACTORY.CONFIG_CLASS(output_activation="sigmoid")
        self.model: tf.keras.Model = model_factory.create_model(input_size=np.product(observation_spec.shape) + latent_size, output_size=np.product(action_spec.shape), config=config)
        return self.model

    def preprocess_observations(self, observations: tf.Tensor) -> tf.Tensor:
        observations -= self.observation_range[0]
        observations /= np.diff(self.observation_range, axis=0).squeeze(0)
        return observations * 2 - 1

    def postprocess_actions(self, actions: tf.Tensor) -> tf.Tensor:
        actions = tf.reshape(actions, shape=(-1, *self.action_shape))
        actions *= np.diff(self.action_range, axis=0).squeeze(0)
        actions += self.action_range[0]
        return actions

    def normalise_outputs(self, actions: np.ndarray) -> np.ndarray:
        actions -= self.action_range[0]
        actions /= np.diff(self.action_range, axis=0).squeeze(0)
        return actions

    def add_noise_to_observations(self, observations: np.ndarray, mu: float = 1.) -> np.ndarray:
        if mu == 0:
            noise = np.zeros((len(observations), self.latent_size))
        else:
            noise = np.random.randn(len(observations), self.latent_size) * mu
        observations = np.concatenate((noise, observations), axis=1)
        return observations

    def generate_actions(self, observation: np.ndarray, n: Optional[int] = None) -> np.ndarray:
        batched_throughput = True
        if n is None:
            batched_throughput = False
            n = 1

        batch = np.tile(observation, (n, 1))
        batch = self.preprocess_observations(batch)
        input = self.add_noise_to_observations(batch)

        samples = self.model.predict(input, batch_size=256, verbose=0)
        samples = self.postprocess_actions(samples)

        if not batched_throughput:
            samples = tf.squeeze(samples, 0)

        samples = samples.numpy()
        return samples

    def learn(
        self,
        training_history: List[Tuple[int, np.ndarray, np.ndarray, float]],
        augmentation_function: Callable[[np.ndarray, np.ndarray, float], List[Tuple[np.ndarray, np.ndarray, float]]],
        training_config: TrainingConfig = TrainingConfig()
    ) -> callbacks.History:
        training_data = [(augmented_observation, augmented_action) for (player, observation, action, reward) in training_history for (augmented_observation, augmented_action, augmented_reward) in (augmentation_function(observation, action, reward) if np.sign(player) == np.sign(reward) else [])]
        observations, actions = zip(*training_data)

        observations = self.add_noise_to_observations(observations)

        return self.fit(observations, actions, training_config)
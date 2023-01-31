from src.sampling.base import SamplingStrategy
from src.model import ModelFactory, BEST_MODEL_FACTORY
from src.io import ModelDecorator

import tensorflow as tf
import numpy as np

from typing import Callable, List, Optional, Tuple
from dm_env.specs import BoundedArray

class NNSamplingStrategy(SamplingStrategy, ModelDecorator):
    DEFAULT_MODEL_FILE = "sampler.h5"
    def __init__(self, action_spec: BoundedArray, observation_spec: BoundedArray, model_factory: ModelFactory = BEST_MODEL_FACTORY, latent_size: int = 4):
        super().__init__(action_spec, observation_spec)
        self.latent_size = latent_size
        self.setup_model(action_spec, observation_spec, model_factory, latent_size)

    def setup_model(self, action_spec, observation_spec, model_factory, latent_size):
        config = BEST_MODEL_FACTORY.CONFIG_CLASS(output_activation="sigmoid")
        self.model: tf.keras.Model = model_factory.create_model(input_size=np.product(observation_spec.shape) + latent_size, output_size=np.product(action_spec.shape), config=config)

    @tf.function
    def postprocess_actions(self, actions: tf.Tensor) -> tf.Tensor:
        actions *= self.action_range[1] - self.action_range[0]
        actions += self.action_range[0]
        return actions

    def normalise_outputs(self, actions: np.ndarray) -> np.ndarray:
        actions -= self.action_range[0]
        actions /= self.action_range[1] - self.action_range[0]
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
        input = self.add_noise_to_observations(batch)

        samples = self.model.predict(input, batch_size=256, verbose=0)
        samples = tf.reshape(samples, shape=(-1, *self.action_shape))
        if not batched_throughput:
            samples = tf.squeeze(samples, 0)

        samples = self.postprocess_actions(samples)
        samples = samples.numpy()
        return samples

    def learn(self, training_history: List[Tuple[int, np.ndarray, np.ndarray, float]], augmentation_function: Callable[[np.ndarray, np.ndarray, float], List[Tuple[np.ndarray, np.ndarray, float]]], **hyperparams):
        training_data = [(augmented_observation, augmented_action) for (player, observation, action, reward) in training_history for (augmented_observation, augmented_action, augmented_reward) in (augmentation_function(observation, action, reward) if np.sign(player) == np.sign(reward) else [])]
        observations, actions = zip(*training_data)

        observations = self.add_noise_to_observations(observations)

        self.fit(observations, np.array(actions), **hyperparams)
        return

class WeightedNNSamplingStrategy(NNSamplingStrategy):
    def learn(self, training_history: List[Tuple[int, np.ndarray, np.ndarray, float]], augmentation_function: Callable[[np.ndarray, np.ndarray, float], List[Tuple[np.ndarray, np.ndarray, float]]], **hyperparams):
        training_data = [(augmented_observation, augmented_action, reward * np.sign(player) * np.sign(reward)) for (player, observation, action, reward) in training_history for (augmented_observation, augmented_action, augmented_reward) in (augmentation_function(observation, action, reward))]
        observations, actions, weights = zip(*training_data)

        observations = self.add_noise_to_observations(observations)

        # increase patience
        hyperparams["patience"] = hyperparams.get("epochs", 1000)
        hyperparams["validation_split"] = 0
        self.fit(observations, np.array(actions), sample_weight=np.array(weights), **hyperparams)
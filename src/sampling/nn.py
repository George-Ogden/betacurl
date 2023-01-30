from src.sampling.base import SamplingStrategy
from src.model import ModelFactory, BEST_MODEL_FACTORY
from src.io import ModelDecorator

from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np

from tqdm.keras import TqdmCallback
import wandb

from typing import Callable, List, Optional, Tuple
from dm_env.specs import BoundedArray

class NNSamplingStrategy(SamplingStrategy, ModelDecorator):
    DEFAULT_MODEL_FILE = "sampler.h5"
    def __init__(self, action_spec: BoundedArray, observation_spec: BoundedArray, model_factory: ModelFactory = BEST_MODEL_FACTORY, latent_size: int = 4):
        super().__init__(action_spec, observation_spec)
        self.latent_size = latent_size
        config = BEST_MODEL_FACTORY.CONFIG_CLASS(output_activation="sigmoid")
        self.model: tf.keras.Model = model_factory.create_model(input_size=np.product(observation_spec.shape) + latent_size, output_size=np.product(action_spec.shape), config=config)

    def postprocess_actions(self, actions: np.ndarray) -> np.ndarray:
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

        samples = samples.numpy()
        samples = self.postprocess_actions(samples)
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

class WeightedNNSamplingStrategy(NNSamplingStrategy):
    def compute_loss(self, batch: np.ndarray, actions: tf.Tensor, rewards: tf.Tensor) -> tf.Tensor:
        mean, log_std = self.model(batch)
        log_probs = self.compute_log_probs(actions, mean, log_std)
        loss = -tf.reduce_mean(log_probs * rewards)
        return loss
    
    @tf.function
    def train_step(self, batch: np.ndarray, optimizer: tf.optimizers.Optimizer):
        with tf.GradientTape() as tape:
            observations, actions, rewards = batch
            loss = self.compute_loss(observations, actions, rewards)
        grads = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss
    
    def compute_validation_loss(self, validation_data: np.ndarray) -> tf.Tensor:
        validation_loss = 0
        for batch in validation_data:
            actions = batch[1]
            rewards = batch[2]
            validation_loss += self.compute_loss(batch[0], actions, rewards)
        return validation_loss / len(validation_data)


    def learn(self, training_history: List[Tuple[int, np.ndarray, np.ndarray, float]], augmentation_function: Callable[[np.ndarray, np.ndarray, float], List[Tuple[np.ndarray, np.ndarray, float]]], **hyperparams):
        training_data = [(augmented_observation, augmented_action, reward * np.sign(player) * np.sign(reward)) for (player, observation, action, reward) in training_history for (augmented_observation, augmented_action, augmented_reward) in (augmentation_function(observation, action, reward))]
        training_data, validation_data = train_test_split(training_data, test_size=hyperparams.get("validation_split"))
        

        optimizer = tf.optimizers.Adam(learning_rate=hyperparams.get("learning_rate", 1e-2))
        history = tf.keras.callbacks.History()
        callbacks = [callbacks.EarlyStopping(patience=hyperparams.get("patience", 5), monitor="val_loss"), WandbMetricsLogger(), history]
        batch_size = hyperparams.get("batch_size", 16)
        for epoch in range(hyperparams.get("epochs", 20)):
            # TODO: shuffle data
            for i in range(0, len(training_data), batch_size):
                batch = training_data[i:i + batch_size]
                loss = self.train_step(batch, optimizer)
            validation_loss = self.compute_validation_loss(validation_data)
            for callback in callbacks:
                callback.on_epoch_end(epoch, {"loss": loss, "val_loss": validation_loss})
        return history


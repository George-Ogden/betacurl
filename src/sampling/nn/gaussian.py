from sklearn.model_selection import train_test_split
from tensorflow_probability import distributions
from tensorflow.keras import callbacks
import tensorflow as tf
import numpy as np


from typing import Callable, List, Tuple
from dm_env.specs import BoundedArray

from .nn import NNSamplingStrategy
from ...model import ModelFactory, TrainingConfig, BEST_MODEL_FACTORY

class GaussianSamplingStrategy(NNSamplingStrategy):
    def __init__(self, action_spec: BoundedArray, observation_spec: BoundedArray, model_factory: ModelFactory = BEST_MODEL_FACTORY, latent_size: int = 4):
        super().__init__(action_spec, observation_spec, model_factory, latent_size)
        self.action_mean = (self.action_range[0] + self.action_range[1]) / 2
    
    def setup_model(self, action_spec, observation_spec, model_factory, latent_size):
        config = BEST_MODEL_FACTORY.CONFIG_CLASS(output_activation="linear")
        self.model: tf.keras.Model = model_factory.create_model(input_size=np.product(observation_spec.shape) + latent_size, output_size=np.product(action_spec.shape) * 2, config=config)
    
    def add_noise_to_observations(self, observations: np.ndarray, mu: float = 0) -> np.ndarray:
        # use distribution rather than sampling from noise
        return observations
    
    @tf.function
    def postprocess_actions(self, actions: tf.Tensor) -> tf.Tensor:
        actions = tf.reshape(actions, (actions.shape[0], *self.action_shape, 2))
        means, log_stds = tf.split(actions, 2, axis=-1)
        
        means += self.action_mean
        stds = tf.exp(log_stds)
        normal = distributions.Normal(means, stds)
        
        return normal.sample()

    @staticmethod
    def compute_log_probs(means, log_stds, actions):
        normal = distributions.Normal(means, tf.exp(log_stds))
        log_probs = normal.log_prob(actions)
        return tf.reduce_sum(log_probs, axis=-1)

    def compute_loss(self, batch: np.ndarray, actions: tf.Tensor, rewards: tf.Tensor) -> tf.Tensor:
        mean, log_std = tf.split(self.model(batch), 2, axis=-1)
        log_probs = self.compute_log_probs(mean, log_std, actions)
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
            observations, actions, rewards = batch
            validation_loss += self.compute_loss(observations, actions, rewards)
        return validation_loss / len(validation_data)


    def learn(
        self,
        training_history: List[Tuple[int, np.ndarray, np.ndarray, float]],
        augmentation_function: Callable[[np.ndarray, np.ndarray, float], List[Tuple[np.ndarray, np.ndarray, float]]],
        training_config: TrainingConfig = TrainingConfig()
    ) -> callbacks.History:
        training_data = [(augmented_observation, augmented_action, reward * np.sign(player) * np.sign(reward)) for (player, observation, action, reward) in training_history for (augmented_observation, augmented_action, augmented_reward) in (augmentation_function(observation, action, reward))]
        training_data, validation_data = train_test_split(training_data, test_size=training_config.validation_split)

        self.compile_model(training_config)
        train_dataset = self.create_dataset(training_data)
        val_dataset = self.create_dataset(validation_data)
        
        optimizer = training_config.optimizer
        history = callbacks.History()
        callbacks = training_config.callbacks + [history]
        batch_size = training_config.batch_size
        for epoch in training_config.epochs:
            # TODO: shuffle data
            for i in range(0, len(training_data), batch_size):
                batch = training_data[i:i + batch_size]
                loss = self.train_step(batch, optimizer)
            validation_loss = self.compute_validation_loss(validation_data)
            for callback in callbacks:
                callback.on_epoch_end(epoch, {"loss": loss, "val_loss": validation_loss})
        return history
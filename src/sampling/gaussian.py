from src.sampling.nn import NNSamplingStrategy
from src.model import ModelFactory, BEST_MODEL_FACTORY

from sklearn.model_selection import train_test_split
from tensorflow_probability import distributions
import tensorflow as tf
import numpy as np

from wandb.keras import WandbMetricsLogger
from tensorflow.keras import callbacks
from tqdm.keras import TqdmCallback

from typing import Callable, List, Tuple

from dm_env.specs import BoundedArray

class GaussianSamplingStrategy(NNSamplingStrategy):
    def __init__(self, action_spec: BoundedArray, observation_spec: BoundedArray, model_factory: ModelFactory = BEST_MODEL_FACTORY, latent_size: int = 4):
        super().__init__(action_spec, observation_spec, model_factory, latent_size)
        self.action_mean = (self.action_range[0] + self.action_range[1]) / 2
    
    def setup_model(self, action_spec, observation_spec, model_factory, latent_size):
        config = BEST_MODEL_FACTORY.CONFIG_CLASS(output_activation="linear")
        self.model: tf.keras.Model = model_factory.create_model(input_size=np.product(observation_spec.shape) + latent_size, output_size=np.product(action_spec.shape) * 2, config=config)
    
    @tf.function
    def postprocess_actions(self, actions: tf.Tensor) -> tf.Tensor:
        actions = tf.reshape(actions, (actions.shape[0], *self.action_shape, 2))
        means, log_stds = tf.split(actions, 2, axis=-1)
        
        means += self.action_mean
        stds = tf.exp(log_stds)
        normal = distributions.Normal(means, stds)
        
        return normal.sample()

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
            observations, actions, rewards = batch
            validation_loss += self.compute_loss(observations, actions, rewards)
        return validation_loss / len(validation_data)


    def learn(self, training_history: List[Tuple[int, np.ndarray, np.ndarray, float]], augmentation_function: Callable[[np.ndarray, np.ndarray, float], List[Tuple[np.ndarray, np.ndarray, float]]], **hyperparams) -> callbacks.History:
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


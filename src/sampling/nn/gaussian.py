from tensorflow_probability import distributions
from tensorflow.keras import callbacks, utils
from copy import copy, deepcopy
from tensorflow import data
import tensorflow as tf
import numpy as np


from typing import Callable, List, Tuple
from dm_env.specs import BoundedArray

from ...model import ModelFactory, TrainingConfig, BEST_MODEL_FACTORY
from .nn import NNSamplingStrategy

class GaussianSamplingStrategy(NNSamplingStrategy):
    epsilon = .1
    def __init__(self, action_spec: BoundedArray, observation_spec: BoundedArray, model_factory: ModelFactory = BEST_MODEL_FACTORY):
        super().__init__(action_spec, observation_spec, model_factory, latent_size=0)
        self.action_mean = (self.action_range[0] + self.action_range[1]) / 2
    
    def setup_model(self, action_spec, observation_spec, model_factory, latent_size=0):
        config = model_factory.CONFIG_CLASS(output_activation="linear")
        self.model: tf.keras.Model = model_factory.create_model(input_size=np.product(observation_spec.shape) + latent_size, output_size=np.product(action_spec.shape) * 2, config=config)
        self.target_model = None
    
    def add_noise_to_observations(self, observations: np.ndarray, mu: float = 0) -> np.ndarray:
        # use distribution rather than sampling from noise
        return observations

    @tf.function
    def generate_distribution(self, raw_actions: tf.Tensor) -> distributions.Distribution:
        actions = tf.reshape(raw_actions, (raw_actions.shape[0], *self.action_shape, 2))
        means, log_stds = tf.split(actions, 2, axis=-1)
        means = tf.squeeze(means, -1)
        log_stds = tf.squeeze(log_stds, -1)
        
        means += self.action_mean
        stds = tf.exp(log_stds)
        return distributions.Normal(means, stds)
    
    @tf.function
    def postprocess_actions(self, actions: tf.Tensor) -> tf.Tensor:
        normal = self.generate_distribution(actions)
        return tf.clip_by_value(normal.sample(), *self.action_range)

    @staticmethod
    @tf.function
    def compute_log_probs(distribution: distributions.Distribution, actions: tf.Tensor) -> tf.Tensor:
        log_probs = distribution.log_prob(actions)
        return tf.reduce_sum(log_probs, axis=-1)

    @tf.function
    def compute_loss(self, observations: np.ndarray, actions: tf.Tensor, rewards: tf.Tensor) -> tf.Tensor:
        predicted_distribution = self.generate_distribution(
            self.model(observations)
        )
        target_distribution = self.generate_distribution(
            self.target_model(observations)
        )

        log_probs = self.compute_log_probs(predicted_distribution, actions)
        target_log_probs = self.compute_log_probs(target_distribution, actions)
        ratio = tf.exp(log_probs - target_log_probs)

        loss_1 = rewards * ratio
        loss_2 = rewards * tf.clip_by_value(ratio, 1 - self.epsilon, 1 + self.epsilon)
        loss = tf.math.minimum(loss_1, loss_2)

        return -tf.reduce_mean(loss)

    def train_step(self, batch: np.ndarray, optimizer: tf.optimizers.Optimizer):
        with tf.GradientTape() as tape:
            observations, actions, rewards = batch
            loss = self.compute_loss(observations, actions, rewards)
        grads = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def fit(self, dataset: data.Dataset, training_config: TrainingConfig = TrainingConfig()):
        training_config = copy(training_config)
        train_dataset, val_dataset = utils.split_dataset(dataset, right_size=training_config.validation_split, shuffle=True)
        optimizer = training_config.optimizer
        batch_size = training_config.batch_size
        training_config.metrics = []

        history = callbacks.History()
        callback = callbacks.CallbackList(
            training_config.callbacks + [history],
            model=self.model,
            add_history=False,
            add_progbar=training_config.verbose != 0,
            verbose=training_config.verbose,
            epochs=training_config.training_epochs,
            steps=len(train_dataset) // training_config.batch_size,
        )

        callback.on_train_begin()
        for epoch in range(training_config.training_epochs):
            callback.on_epoch_begin(epoch)
            loss = 0
            for step, batch in enumerate(train_dataset.batch(batch_size)):
                callback.on_train_batch_begin(step)
                loss += self.train_step(batch, optimizer)
                callback.on_train_batch_end(step)
            val_loss = 0
            for step, batch in enumerate(val_dataset.batch(batch_size)):
                val_loss += self.compute_loss(*batch)
            callback.on_epoch_end(epoch, {"loss": loss / len(train_dataset), "val_loss": val_loss / len(val_dataset)})
            if self.model.stop_training:
                break
        callback.on_train_end()
        return history

    def learn(
        self,
        training_history: List[Tuple[int, np.ndarray, np.ndarray, float]],
        augmentation_function: Callable[[np.ndarray, np.ndarray, float], List[Tuple[np.ndarray, np.ndarray, float]]],
        training_config: TrainingConfig = TrainingConfig()
    ) -> callbacks.History:
        training_data = [(augmented_observation, augmented_action, reward * np.sign(player) * np.sign(reward)) for (player, observation, action, reward) in training_history for (augmented_observation, augmented_action, augmented_reward) in (augmentation_function(observation, action, reward))]

        self.target_model = deepcopy(self.model)
        self.compile_model(training_config)
        dataset = self.create_dataset(training_data)
        
        return self.fit(dataset, training_config)
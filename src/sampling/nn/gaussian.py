from tensorflow_probability import distributions
from tensorflow.keras import callbacks, models, utils
from copy import copy
from tensorflow import data
import tensorflow as tf
import numpy as np


from typing import Callable, List, Tuple
from dm_env.specs import BoundedArray

from ...model import ModelFactory, TrainingConfig, BEST_MODEL_FACTORY
from ..config import GaussianSamplerConfig
from .nn import NNSamplingStrategy

class GaussianSamplingStrategy(NNSamplingStrategy):
    CONFIG_CLASS = GaussianSamplerConfig
    def __init__(self, action_spec: BoundedArray, observation_spec: BoundedArray, model_factory: ModelFactory = BEST_MODEL_FACTORY, config: GaussianSamplerConfig = GaussianSamplerConfig()):
        super().__init__(action_spec, observation_spec, model_factory, config)
        self.action_mean = (self.action_range[0] + self.action_range[1]) / 2
        
        assert config.clip_ratio > 0
        self.clip_ratio = config.clip_ratio
        self.max_grad_norm = config.max_grad_norm
    
    def setup_model(self, action_spec: BoundedArray, observation_spec: BoundedArray, model_factory: ModelFactory, latent_size: int = 0)  -> tf.keras.Model:
        config = model_factory.CONFIG_CLASS(output_activation="linear")
        self.model: tf.keras.Model = model_factory.create_model(input_size=np.product(observation_spec.shape) + latent_size, output_size=np.product(action_spec.shape) * 2, config=config)
        return self.model
    
    def add_noise_to_observations(self, observations: np.ndarray, mu: float = 0) -> np.ndarray:
        # use distribution rather than sampling from noise
        return observations

    def generate_distribution(self, raw_actions: tf.Tensor) -> distributions.Distribution:
        actions = tf.reshape(raw_actions, (raw_actions.shape[0], *self.action_shape, 2))
        means, log_stds = tf.split(actions, 2, axis=-1)
        means = tf.squeeze(means, -1)
        log_stds = tf.squeeze(log_stds, -1)
        
        means += self.action_mean
        stds = tf.exp(log_stds)
        return distributions.TruncatedNormal(means, stds, *self.action_range)

    def postprocess_actions(self, actions: tf.Tensor) -> tf.Tensor:
        distribution = self.generate_distribution(actions)
        return distribution.sample()

    @staticmethod
    def compute_log_probs(distribution: distributions.Distribution, actions: tf.Tensor) -> tf.Tensor:
        log_probs = distribution.log_prob(actions)
        return tf.reduce_sum(log_probs, axis=-1)

    def compute_loss(self, observations: np.ndarray, actions: tf.Tensor, values: tf.Tensor, advantage: tf.Tensor, target_log_probs: tf.Tensor) -> tf.Tensor:
        predicted_distribution = self.generate_distribution(
            self.model(
                self.preprocess_observations(
                    observations
                )
            )
        )
        
        return self.ppo_clip_loss(
            log_probs=self.compute_log_probs(predicted_distribution, actions),
            target_log_probs=target_log_probs,
            advantages=advantage
        )

    def ppo_clip_loss(
        self,
        log_probs: tf.Tensor,
        target_log_probs: tf.Tensor,
        advantages: tf.Tensor
    ) -> tf.Tensor:
        ratio = tf.exp(log_probs - target_log_probs)

        loss_1 = advantages * ratio
        loss_2 = advantages * tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
        loss = tf.math.minimum(loss_1, loss_2)

        return -tf.reduce_mean(loss)

    def train_step(self, batch: np.ndarray, optimizer: tf.optimizers.Optimizer) -> np.ndarray:
        with tf.GradientTape() as tape:
            loss = self.compute_loss(*batch)
        grads = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def fit(self, dataset: data.Dataset, training_config: TrainingConfig = TrainingConfig()) -> callbacks.History:
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
            steps=(len(train_dataset) - 1) // training_config.batch_size + 1,
        )

        self.model.stop_training = False
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

    def compute_advantages_and_target_log_probs(
        self,
        players: tf.Tensor,
        observations: tf.Tensor,
        actions: tf.Tensor,
        rewards: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        advantages = tf.sign(players) * rewards

        target_distribution = self.generate_distribution(
            self.target_model(
                self.preprocess_observations(
                    observations
                )
            )
        )
    
        target_log_probs = self.compute_log_probs(
            target_distribution,
            actions
        )
        return advantages, target_log_probs

    def learn(
        self,
        training_history: List[Tuple[int, np.ndarray, np.ndarray, float]],
        augmentation_function: Callable[[int, np.ndarray, np.ndarray, float], List[Tuple[int, np.ndarray, np.ndarray, float]]],
        training_config: TrainingConfig = TrainingConfig()
    ) -> callbacks.History:
        self.target_model = models.clone_model(self.model)

        training_data = [
            (augmented_player,
            augmented_observation,
            augmented_action,
            augmented_reward
        ) for (player, observation, action, reward) in training_history 
            for (augmented_player, augmented_observation, augmented_action, augmented_reward) in (augmentation_function(player, observation, action, reward))]
        primary_dataset = self.create_dataset(training_data).batch(training_config.batch_size)
        
        batched_transform = [(observation, action, reward) + self.compute_advantages_and_target_log_probs(player, observation, action, reward) for player, observation, action, reward in primary_dataset]
        flattened_transform = [np.concatenate(data, axis=0) for data in zip(*batched_transform)]
        secondary_dataset = self.create_dataset(zip(*flattened_transform))

        if training_config.optimizer_kwargs is None:
            training_config.optimizer_kwargs = {"clipnorm": self.max_grad_norm}
        else:
            training_config.optimizer_kwargs["clipnorm"] = self.max_grad_norm

        self.compile_model(training_config)
        
        return self.fit(secondary_dataset, training_config)
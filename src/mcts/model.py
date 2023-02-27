from tensorflow.keras import callbacks, layers, losses, models
from tensorflow_probability import distributions
from tensorflow import keras
import tensorflow as tf
from copy import copy
import numpy as np

from typing import Callable, List, Optional, Tuple, Union

from ..model import CustomDecorator, DenseModelFactory, ModelFactory, TrainingConfig, BEST_MODEL_FACTORY
from ..utils import SaveableMultiModel

from .config import MCTSModelConfig

class MCTSModel(SaveableMultiModel, CustomDecorator):
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

        self.config = copy(config)
        self.feature_size = config.feature_size
        self.max_grad_norm = config.max_grad_norm
        self.vf_coeff = config.vf_coeff
        self.ent_coeff = config.ent_coeff

        self.feature_extractor = model_factory.create_model(
            input_shape=self.observation_shape,
            output_shape=self.feature_size,
            config=model_factory.CONFIG_CLASS(
                output_activation="linear"
            )
        )
        # self.feature_extractor = keras.Sequential(
        #     [
        #         layers.Rescaling(
        #             scale=1./np.diff(self.observation_range, axis=0).squeeze(0),
        #             offset=-self.observation_range.mean(axis=0),
        #         ),
        #         self.feature_extractor
        #     ]
        # )

        self.policy_head = DenseModelFactory.create_model(
            input_shape=self.feature_size,
            output_shape=self.action_shape + (2,),
            config=DenseModelFactory.CONFIG_CLASS(
                output_activation="linear"
            )
        )

        self.value_head = DenseModelFactory.create_model(
            input_shape=self.feature_size,
            output_shape=(),
            config=DenseModelFactory.CONFIG_CLASS(
                output_activation="linear"
            )
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
            output_shape=self.feature_size,
            config=DenseModelFactory.CONFIG_CLASS(
                output_activation="linear"
            )
        )(inputs) # not used yet

        if scaling_spec is None:
            self.scaling_spec = np.stack(
                (self.action_range.mean(axis=0), np.zeros(self.action_shape)),
                axis=-1
            )
        elif scaling_spec.ndim == 1:
            self.scaling_spec = np.stack(
                (scaling_spec, np.zeros(self.action_shape)),
                axis=-1
            )
        else:
            self.scaling_spec = scaling_spec.copy()
            self.scaling_spec[:, 1] = np.log(scaling_spec[:, 1])

        assert self.scaling_spec.shape == self.action_shape + (2,)

        self.policy_head = keras.Sequential(
            [self.policy_head, layers.Rescaling(offset=self.scaling_spec, scale=1.)]
        )

        input = keras.Input(self.observation_shape)
        self.model = models.Model(
            inputs=input,
            outputs=[
                self.policy_head(self.feature_extractor(input)),
                self.value_head(self.feature_extractor(input)),
            ]
        )

    def save(self, directory: str):
        SaveableMultiModel.save(self, directory)
        CustomDecorator.save(self, directory)

    @classmethod
    def load(cls, directory: str) -> "Self":
        model = SaveableMultiModel.load(directory)
        model.model = models.load_model(model.get_model_filename(directory))

    def predict_values(self, observation: Union[tf.Tensor, np.ndarray], training: bool=False) -> Union[tf.Tensor, np.ndarray]:
        batch_throughput = True
        if observation.ndim == len(self.observation_shape):
            batch_throughput = False
            observation = np.expand_dims(observation, 0)

        features = self.feature_extractor(observation, training=training)
        values = self.value_head(features, training=training)

        if not batch_throughput:
            values = tf.squeeze(values, 0).numpy()

        return values

    def compute_advantages(
        self,
        players: tf.Tensor,
        observations: tf.Tensor,
        rewards: tf.Tensor
    ) -> tf.Tensor:
        value_predictions = self.predict_values(observations, training=False)
        advantages = rewards - value_predictions
        return advantages * tf.sign(players)

    def compute_loss(self,
        observations: np.ndarray,
        actions: tf.Tensor,
        values: tf.Tensor,
        advantages: tf.Tensor
    ) -> tf.Tensor:
        predicted_distribution = self.generate_distribution(observations, training=True)
        predicted_values = self.predict_values(observations, training=True)

        policy_loss = -tf.reduce_mean(advantages * tf.reduce_sum(predicted_distribution.log_prob(actions), axis=-1))
        value_loss = losses.mean_squared_error(values, predicted_values)
        loss = policy_loss + self.vf_coeff * value_loss
        if self.ent_coeff != 0:
            # entropy is main cause of NaNs in training
            entropy_loss = -tf.reduce_mean(predicted_distribution.entropy())
            loss += self.ent_coeff * entropy_loss
        return loss

    def generate_distribution(self, observation: Union[tf.Tensor, np.ndarray], training: bool=False) -> distributions.Distribution:
        batch_throughput = True
        if observation.ndim == len(self.observation_shape):
            batch_throughput = False
            observation = np.expand_dims(observation, 0)

        features = self.feature_extractor(observation, training=training)
        raw_actions = self.policy_head(features, training=training)

        if not batch_throughput:
            raw_actions = tf.squeeze(raw_actions, 0).numpy()

        means, log_stds = tf.split(raw_actions, 2, axis=-1)
        means = tf.squeeze(means, -1)
        log_stds = tf.squeeze(log_stds, -1)
        stds = tf.exp(log_stds)

        return distributions.TruncatedNormal(means, stds, *self.action_range)
    
    def learn(
        self,
        training_history: List[Tuple[int, np.ndarray, np.ndarray, float]],
        augmentation_function: Callable[[int, np.ndarray, np.ndarray, float], List[Tuple[int, np.ndarray, np.ndarray, float]]],
        training_config: TrainingConfig = TrainingConfig()
    ) -> callbacks.History:
        training_data = [
            (augmented_player,
            augmented_observation,
            augmented_action,
            augmented_reward
        ) for (player, observation, action, reward) in training_history 
            for (augmented_player, augmented_observation, augmented_action, augmented_reward) in (augmentation_function(player, observation, action, reward))]
        primary_dataset = self.create_dataset(training_data).batch(training_config.batch_size)

        batched_transform = [(observation, action, reward) + (self.compute_advantages(players=player, observations=observation, rewards=reward),) for player, observation, action, reward in primary_dataset]
        flattened_transform = [np.concatenate(data, axis=0) for data in zip(*batched_transform)]
        secondary_dataset = self.create_dataset(zip(*flattened_transform))

        training_config.optimizer_kwargs["clipnorm"] = self.max_grad_norm
        
        return self.fit(secondary_dataset, training_config)
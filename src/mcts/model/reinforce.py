from tensorflow_probability import distributions
from tensorflow.keras import layers, losses
from tensorflow import data, keras
import tensorflow as tf
import numpy as np

from typing import Callable, List, Optional, Tuple, Union

from ...model import DenseModelFactory, ModelFactory, TrainingConfig, BEST_MODEL_FACTORY
from ...game import GameSpec

from .config import ReinforceMCTSModelConfig
from .fourier import FourierDistribution
from .base import MCTSModel

class ReinforceMCTSModel(MCTSModel):
    MODELS = {
        "feature_extractor": "feature_extractor.h5",
        "policy_head": "policy.h5",
        "value_head": "value.h5"
    }
    def __init__(
        self,
        game_spec: GameSpec,
        model_factory: ModelFactory = BEST_MODEL_FACTORY,
        config: ReinforceMCTSModelConfig = ReinforceMCTSModelConfig()
    ):
        super().__init__(
            game_spec=game_spec,
            config=config
        )
        self.ent_coeff = config.ent_coeff
        self.clip_range = config.clip_range

        self.feature_extractor = model_factory.create_model(
            input_shape=self.observation_shape,
            output_shape=self.feature_size,
            config=model_factory.CONFIG_CLASS(
                output_activation="linear"
            )
        )
        if self.observation_range is not None:
            self.feature_extractor = keras.Sequential(
                [
                    layers.Rescaling(
                        scale=2./np.diff(self.observation_range, axis=0).squeeze(0),
                        offset=-self.observation_range.mean(axis=0),
                    ),
                    self.feature_extractor
                ]
            )

        self.policy_head = DenseModelFactory.create_model(
            input_shape=self.feature_size,
            output_shape=self.action_shape + (config.fourier_features, 2),
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

        self.setup_model()

    def setup_model(self):
        input = keras.Input(self.observation_shape)
        features = self.feature_extractor(input)
        self.model = keras.Model(
            inputs=input,
            outputs=[
                self.policy_head(features),
                self.value_head(features),
            ]
        )
        self.model(np.random.randn(1, *self.observation_shape))

    def predict_values(
        self,
        observation: Union[tf.Tensor, np.ndarray],
        training: bool=False
    ) -> Union[tf.Tensor, np.ndarray]:
        batch_throughput = True
        if observation.ndim == len(self.observation_shape):
            batch_throughput = False
            observation = np.expand_dims(observation, 0)

        features = self.feature_extractor(observation, training=training)
        values = self.value_head(features, training=training)

        if not batch_throughput:
            values = tf.squeeze(values, 0).numpy()

        return values

    def compute_loss(
        self,
        observations: tf.Tensor,
        action_groups: tf.RaggedTensor,
        values: tf.Tensor,
        advantage_groups: tf.RaggedTensor
    ) -> tf.Tensor:
        predicted_distribution = self.generate_distribution(observations, training=True)
        predicted_values = self.predict_values(observations, training=True)

        if isinstance(advantage_groups, tf.RaggedTensor) or isinstance(action_groups, tf.RaggedTensor):
            policy_loss = 0.
            clip_fraction = 0.

            for i, (actions, advantages) in enumerate(zip(action_groups, advantage_groups, strict=True)):
                distribution = predicted_distribution[i]
                other_dims = tuple(range(1, distribution.batch_shape.ndims))

                log_probs = distribution.log_prob(actions.to_tensor())
                clipped_log_probs = tf.clip_by_value(log_probs, -self.clip_range, self.clip_range)

                policy_loss -= tf.reduce_mean(advantages * tf.reduce_sum(clipped_log_probs, axis=-1))

                clip_fraction += tf.reduce_mean(
                    tf.cast(
                        tf.greater(tf.abs(log_probs), self.clip_range),
                        tf.float32
                    )
                ).numpy()
            policy_loss /= action_groups.shape[0]
            clip_fraction /= action_groups.shape[0]
        else:
            log_probs = predicted_distribution.log_prob(tf.transpose(action_groups, (1, 0, *range(2, action_groups.ndim))))
            advantage_groups = tf.transpose(advantage_groups, (1, 0))
            other_dims = tuple(range(2, predicted_distribution.batch_shape.ndims))

            clipped_log_probs = tf.clip_by_value(log_probs, -self.clip_range, self.clip_range)
            policy_loss = -tf.reduce_mean(
                tf.transpose(
                    tf.reduce_sum(clipped_log_probs, axis=(other_dims)),
                    (2, 0, 1)
                ) * advantage_groups
            )

            clip_fraction = tf.reduce_mean(
                tf.reduce_mean(
                    tf.cast(
                        tf.greater(tf.abs(log_probs), self.clip_range),
                        tf.float32
                    ),
                    axis=0
                )
            ).numpy()

        value_loss = losses.mean_squared_error(values, predicted_values)
        entropy_loss = -tf.reduce_mean(predicted_distribution.entropy())

        loss = policy_loss + self.vf_coeff * value_loss

        if self.ent_coeff != 0:
            # entropy is main cause of NaNs in training
            loss += self.ent_coeff * entropy_loss

        # record values for logging
        self.stats["policy_loss"] += policy_loss.numpy()
        self.stats["entropy_loss"] += entropy_loss.numpy()
        self.stats["value_loss"] += value_loss.numpy()
        self.stats["loss"] += loss.numpy()
        self.stats["clip_fraction"] += clip_fraction
        self.stats["entropy"] += tf.reduce_sum(predicted_distribution.entropy()).numpy()

        return loss

    def generate_distribution(
        self,
        observation: Union[tf.Tensor, np.ndarray],
        training: bool = False
    ) -> distributions.Distribution:
        batch_throughput = True
        if observation.ndim == len(self.observation_shape):
            batch_throughput = False
            observation = np.expand_dims(observation, 0)

        features = self.feature_extractor(observation, training=training)
        raw_actions = self.policy_head(features, training=training)

        if not batch_throughput:
            raw_actions = tf.squeeze(raw_actions, 0)

        return self._generate_distribution(raw_actions)

    def _generate_distribution(self, raw_actions: tf.Tensor) -> distributions.Distribution:
        range_dim = self.action_range.ndim
        action_range = np.transpose(self.action_range, (*range(1, range_dim), 0))
        bounds = np.tile(
            action_range,
            raw_actions.shape[:-range_dim-1] + (1, ) * range_dim
        )
        return FourierDistribution(
            raw_actions,
            bounds=bounds
        )

    def preprocess_data(
        self,
        training_data: List[Tuple[int, np.ndarray, np.ndarray, float, List[Tuple[np.ndarray, float]]]],
        augmentation_function: Callable[[int, np.ndarray, np.ndarray, float], List[Tuple[int, np.ndarray, np.ndarray, float]]],
        training_config: TrainingConfig = TrainingConfig()
    ) -> data.Dataset:
        training_data = [
            (
                augmented_observation,
                augmented_actions,
                augmented_reward,
                advantages
            ) for player, observation, action, reward, policy in training_data
                for (augmented_player, augmented_observation, augmented_action, augmented_reward),
                    (augmented_actions, advantages)
            in zip(
                augmentation_function(player, observation, action, reward),
                [
                    zip(*policy)
                    for policy in zip(*[
                        [
                            (augmented_action, advantage)
                            for augmented_player, augmented_observation, augmented_action, augmented_reward
                            in augmentation_function(player, observation, action, reward)
                        ]
                        for action, advantage in policy
                    ])
                ],
                strict=True
            )
        ]

        training_config.optimizer_kwargs["clipnorm"] = self.max_grad_norm

        return self.create_dataset(training_data)
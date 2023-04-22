from tensorflow_probability import distributions
from tensorflow.keras import layers, losses
from tensorflow import data, keras
import tensorflow as tf
import numpy as np

from typing import Callable, List, Tuple, Union

from ...model import DenseModelFactory, ModelFactory, TrainingConfig, BEST_MODEL_FACTORY
from ...game import GameSpec

from .config import PolicyMCTSModelConfig
from .fourier import FourierDistribution
from .base import MCTSModel

class PolicyMCTSModel(MCTSModel):
    MODELS = {
        "feature_extractor": "feature_extractor.h5",
        "policy_head": "policy.h5",
        "value_head": "value.h5"
    }
    def __init__(
        self,
        game_spec: GameSpec,
        model_factory: ModelFactory = BEST_MODEL_FACTORY,
        config: PolicyMCTSModelConfig = PolicyMCTSModelConfig()
    ):
        super().__init__(
            game_spec=game_spec,
            config=config
        )
        self.ent_coeff = config.ent_coeff

        self.feature_extractor = keras.Sequential([
            layers.BatchNormalization(),
            model_factory.create_model(
                input_shape=self.observation_shape,
                output_shape=self.feature_size,
                config=model_factory.CONFIG_CLASS(
                    output_activation="linear"
                )
            )
        ])

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
        visit_groups: tf.RaggedTensor
    ) -> tf.Tensor:
        predicted_distribution = self.generate_distribution(observations, training=True)
        predicted_values = self.predict_values(observations, training=True)

        if isinstance(visit_groups, tf.RaggedTensor) or isinstance(action_groups, tf.RaggedTensor):
            policy_loss = 0.

            for i, (actions, visits) in enumerate(zip(action_groups, visit_groups, strict=True)):
                distribution = predicted_distribution[i]
                policy = visits / tf.reduce_sum(visits, axis=-1, keepdims=True)
                other_dims = tuple(range(1, distribution.batch_shape.ndims + 1))

                log_probs = distribution.log_prob(actions.to_tensor())
                log_probs = tf.reduce_sum(log_probs, axis=other_dims)
                predicted_policies = tf.math.softmax(log_probs, axis=-1)

                policy_loss += tf.reduce_mean(
                    losses.kld(
                        policy,
                        predicted_policies
                    )
                )
            policy_loss /= action_groups.shape[0]
        else:
            visit_groups = tf.transpose(visit_groups, (1, 0))
            policies = visit_groups / tf.reduce_sum(visit_groups, axis=-1, keepdims=True)
            other_dims = tuple(range(2, predicted_distribution.batch_shape.ndims + 1))

            log_probs = predicted_distribution.log_prob(tf.transpose(action_groups, (1, 0, *range(2, action_groups.ndim))))
            log_probs = tf.reduce_sum(log_probs, axis=other_dims)
            predicted_policies = tf.math.softmax(log_probs, axis=-1)
            policy_loss = tf.reduce_mean(
                losses.kld(
                    policies,
                    predicted_policies
                )
            )

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
                augmented_value,
                visits
            ) for player, observation, action, value, policy in training_data
                for (augmented_player, augmented_observation, augmented_action, augmented_value),
                    (augmented_actions, visits)
            in zip(
                augmentation_function(player, observation, action, value),
                [
                    zip(*policy)
                    for policy in zip(*[
                        [
                            (augmented_action, visits)
                            for augmented_player, augmented_observation, augmented_action, augmented_reward
                            in augmentation_function(player, observation, action, value)
                        ]
                        for action, advantage, visits in policy
                    ])
                ],
                strict=True
            )
        ]

        training_config.optimizer_kwargs["clipnorm"] = self.max_grad_norm

        return self.create_dataset(training_data)
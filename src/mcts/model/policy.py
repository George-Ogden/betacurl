from tensorflow_probability import distributions
from tensorflow.keras import layers, losses
from tensorflow import data, keras
import tensorflow as tf
import numpy as np

from typing import Callable, List, Optional, Tuple, Type, Union

from ...model import DenseModelFactory, ModelFactory, TrainingConfig, BEST_MODEL_FACTORY
from ...distribution import DistributionFactory, CombDistributionFactory
from ...distribution.comb import CombDistribution # remove this import after refactoring
from ...utils import value_to_support
from ...game import GameSpec

from .config import PolicyMCTSModelConfig
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
        config: PolicyMCTSModelConfig = PolicyMCTSModelConfig(),
        DistributionFactory: Optional[Type[DistributionFactory]] = None
    ):
        super().__init__(
            game_spec=game_spec,
            config=config
        )

        if DistributionFactory is None:
            DistributionFactory = CombDistributionFactory

        self.granularity = (config.distribution_config or DistributionFactory.CONFIG_CLASS()).granularity
        self.action_support = CombDistribution.generate_coefficients(
            self.action_range.reshape(2, -1).transpose(1, 0),
            granularity=self.granularity
        )

        self.distibution_factory = DistributionFactory(
            move_spec=game_spec.move_spec,
            config=DistributionFactory.CONFIG_CLASS(
                **(
                    config.distribution_config
                    or {}
                )
            )
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
            output_shape=self.distibution_factory.parameters_shape,
            config=DenseModelFactory.CONFIG_CLASS(
                output_activation="linear"
            )
        )

        self.value_head = DenseModelFactory.create_model(
            input_shape=self.feature_size,
            output_shape=len(self.value_coefficients),
            config=DenseModelFactory.CONFIG_CLASS(
                output_activation="linear"
            )
        )

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
        training: bool=False,
        predict_logits: bool=False
    ) -> Union[tf.Tensor, np.ndarray]:
        batch_throughput = True
        if observation.ndim == len(self.observation_shape):
            batch_throughput = False
            observation = np.expand_dims(observation, 0)

        if predict_logits:
            values = self._predict_value_logits(observation, training)
        else:
            values = self._predict_values(observation, training)

        if not batch_throughput:
            values = tf.squeeze(values, 0).numpy()

        return values

    def _predict_values(self, observation: tf.Tensor, training: bool=False) -> tf.Tensor:
        probabilities = self._predict_value_logits(observation, training)
        values = tf.reduce_sum(probabilities * self.value_coefficients, axis=-1)
        # invert scaling (https://arxiv.org/abs/1805.11593)
        values = self.inverse_scale_values(values)
        return values

    def _predict_value_logits(self, observation: tf.Tensor, training: bool=False) -> tf.Tensor:
        features = self.feature_extractor(observation, training=training)
        value_logits = self.value_head(features, training=training)
        probabilities = tf.nn.softmax(value_logits, axis=-1)
        return probabilities

    def compute_value_loss(self, observations: tf.Tensor, values: tf.Tensor) -> tf.Tensor:
        observations = tf.reshape(observations, (-1, *self.observation_shape))
        values = tf.reshape(values, (-1))
        value_logits = self.values_to_logits(self.scale_values(values))
        predicted_value_logits = self.predict_values(observations, training=True, predict_logits=True)
        return tf.reduce_mean(losses.categorical_crossentropy(value_logits, predicted_value_logits))

    def compute_loss(
        self,
        observations: tf.Tensor,
        distribution: tf.Tensor,
        values: tf.Tensor,
    ) -> tf.Tensor:
        predicted_distribution = self.generate_distribution(observations, training=True)
        policy_loss = tf.reduce_mean(
            losses.categorical_crossentropy(
                tf.reshape(
                    distribution,
                    (-1, self.granularity),
                ),
                predicted_distribution._pdf
            )
        )

        value_loss = self.compute_value_loss(observations, values)
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
        return self.distibution_factory.create_distribution(raw_actions)

    def preprocess_data(
        self,
        training_data: List[Tuple[int, np.ndarray, np.ndarray, float, List[Tuple[np.ndarray, float]]]],
        augmentation_function: Callable[[int, np.ndarray, np.ndarray, float], List[Tuple[int, np.ndarray, np.ndarray, float]]],
        training_config: TrainingConfig = TrainingConfig()
    ) -> data.Dataset:
        training_data = [
            (
                augmented_observation,
                tf.reduce_sum(
                    tf.stack(supports, axis=-1) * visits,
                    axis=-1
                ) / np.sum(visits),
                augmented_value,
            ) for player, observation, action, value, policy in training_data
                for (augmented_player, augmented_observation, augmented_action, augmented_value),
                    (supports, visits)
            in zip(
                augmentation_function(player, observation, action, value),
                [
                    zip(*policy)
                    for policy in zip(*[
                        [
                            (
                                value_to_support(
                                    values=augmented_action.astype(np.float32),
                                    support=self.action_support
                                ),
                                visits
                            )
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
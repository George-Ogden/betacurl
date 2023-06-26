from tensorflow import data
import tensorflow as tf
import numpy as np

from typing import Callable, List, Optional, Tuple, Type

from ...model import ModelFactory, TrainingConfig, BEST_MODEL_FACTORY
from ...distribution import DistributionFactory, NormalSDNDistributionFactory
from ...game import GameSpec

from .config import ReinforceMCTSModelConfig
from .policy import PolicyMCTSModel

class ReinforceMCTSModel(PolicyMCTSModel):
    CONFIG_CLASS = ReinforceMCTSModelConfig
    def __init__(
        self,
        game_spec: GameSpec,
        model_factory: ModelFactory = BEST_MODEL_FACTORY,
        config: Optional[ReinforceMCTSModelConfig] = None,
        DistributionFactory: Optional[Type[DistributionFactory]] = None
    ):
        if DistributionFactory is None:
            DistributionFactory = NormalSDNDistributionFactory

        super().__init__(
            game_spec=game_spec,
            model_factory=model_factory,
            config=config,
            DistributionFactory=DistributionFactory
        )
        self.clip_range = self.config.clip_range

    def compute_loss(
        self,
        observations: tf.Tensor,
        action_groups: tf.RaggedTensor,
        values: tf.Tensor,
        advantage_groups: tf.RaggedTensor
    ) -> tf.Tensor:
        predicted_distribution = self.generate_distribution(observations, training=True)

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
        self.stats["clip_fraction"] += clip_fraction
        self.stats["entropy"] += tf.reduce_sum(predicted_distribution.entropy()).numpy()

        return loss

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
                advantages
            ) for player, observation, action, value, policy in training_data
                for (augmented_player, augmented_observation, augmented_action, augmented_value),
                    (augmented_actions, advantages)
            in zip(
                augmentation_function(player, observation, action, value),
                [
                    zip(*policy)
                    for policy in zip(*[
                        [
                            (augmented_action, advantage)
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
from tensorflow import data
import tensorflow as tf
import numpy as np

from typing import Callable, Optional, List, Tuple, Type

from ...distribution import DistributionFactory, NormalDistributionFactory
from ...model import ModelFactory, TrainingConfig, BEST_MODEL_FACTORY
from ...game import GameSpec

from .config import PPOMCTSModelConfig
from .policy import PolicyMCTSModel

class PPOMCTSModel(PolicyMCTSModel):
    CONFIG_CLASS = PPOMCTSModelConfig
    def __init__(
        self,
        game_spec: GameSpec,
        model_factory: ModelFactory = BEST_MODEL_FACTORY,
        config: Optional[PPOMCTSModelConfig] = None,
        DistributionFactory: Optional[Type[DistributionFactory]] = None
    ):
        if DistributionFactory is None:
            DistributionFactory = NormalDistributionFactory

        super().__init__(
            game_spec,
            model_factory=model_factory,
            config=config,
            DistributionFactory=DistributionFactory
        )
        self.config: PPOMCTSModelConfig
        self.clip_range = self.config.clip_range
        self.target_kl = self.config.target_kl

    def compute_loss(
        self,
        observations: tf.Tensor,
        action_groups: tf.RaggedTensor,
        values: tf.Tensor,
        advantage_groups: tf.RaggedTensor,
        *target_distribution_params: tf.Tensor,
    ) -> tf.Tensor:
        predicted_distribution = self.generate_distribution(observations, training=True)
        target_distribution = type(predicted_distribution)(*target_distribution_params)
        
        if isinstance(advantage_groups, tf.RaggedTensor) or isinstance(action_groups, tf.RaggedTensor):
            policy_loss = 0.
            kl_div = 0.
            clip_fraction = 0.

            distribution_properties = {attr: getattr(predicted_distribution, attr) for attr in predicted_distribution.parameter_properties()}
            target_distribution_properties = {attr: getattr(target_distribution, attr) for attr in target_distribution.parameter_properties()}

            for i, (actions, advantages) in enumerate(zip(action_groups, advantage_groups, strict=True)):
                distribution = type(predicted_distribution)(**{k: v[i] for k, v in distribution_properties.items()})
                target_distribution = type(target_distribution)(**{k: v[i] for k, v in target_distribution_properties.items()})
                other_dims = tuple(range(1, distribution.batch_shape.ndims))
                
                actions = actions.to_tensor()

                log_probs = tf.reduce_sum(distribution.log_prob(actions), axis=-1)
                target_log_probs = tf.reduce_sum(target_distribution.log_prob(actions), axis=-1)
                
                ratio = tf.exp(log_probs - target_log_probs)
                clip_fraction += tf.reduce_mean(
                    tf.cast(
                        tf.greater(tf.abs(ratio - 1), self.clip_range),
                        tf.float32
                    )
                ).numpy()
                
                policy_loss_1 = ratio * advantages
                policy_loss_2 = tf.clip_by_value(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
                policy_loss -= tf.reduce_mean(tf.minimum(policy_loss_1, policy_loss_2))
                
                kl_div += tf.reduce_mean(
                    tf.reduce_sum(
                        target_distribution.kl_divergence(distribution),
                        other_dims
                    )
                )

            policy_loss /= action_groups.shape[0]
            kl_div /= action_groups.shape[0]
            clip_fraction /= action_groups.shape[0]
        else:
            action_groups = tf.transpose(action_groups, (1, 0, *range(2, action_groups.ndim)))
            advantage_groups = tf.transpose(advantage_groups, (1, 0))
            log_probs = tf.reduce_sum(predicted_distribution.log_prob(action_groups), axis=-1)
            target_log_probs = tf.reduce_sum(target_distribution.log_prob(action_groups), axis=-1)
            other_dims = tuple(range(1, predicted_distribution.batch_shape.ndims))

            ratio = tf.exp(log_probs - target_log_probs)
            clip_fraction = tf.reduce_mean(
                tf.cast(
                    tf.greater(tf.abs(ratio - 1), self.clip_range),
                    tf.float32
                )
            ).numpy()

            policy_loss_1 = ratio * advantage_groups
            policy_loss_2 = tf.clip_by_value(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantage_groups
            policy_loss = -tf.reduce_mean(tf.minimum(policy_loss_1, policy_loss_2))

            kl_div = tf.reduce_sum(
                target_distribution.kl_divergence(predicted_distribution),
                other_dims
            )

        value_loss = self.compute_value_loss(observations, values)
        entropy_loss = -tf.reduce_mean(
            tf.reduce_sum(
                predicted_distribution.entropy(),
                axis=range(1, predicted_distribution.batch_shape.ndims)
            )
        )

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
        self.stats["kl_div"] += tf.reduce_sum(kl_div).numpy()
        self.stats["entropy"] += tf.reduce_sum(predicted_distribution.entropy()).numpy()

        # stop early when KL divergence is too high
        if self.target_kl is not None and tf.reduce_mean(kl_div) > self.target_kl:
            self.model.stop_training = True
            loss *= 0.

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


        dataset = self.create_dataset(training_data)

        # generate target distributions from current model
        target_distributions = [
            self.generate_distribution(observation, training=False) 
            for observation, *_ in dataset.batch(training_config.batch_size)
        ]

        # extract parameters for training
        target_distribution_parameters = [
            tf.concat(property, axis=0)
            for property in
            zip(*[
                [
                    getattr(distribution, attr) 
                    for attr in distribution.parameter_properties()
                ]
                for distribution in target_distributions
            ])
        ]

        training_config.optimizer_kwargs["clipnorm"] = self.max_grad_norm
        # zip with dataset
        return self.create_dataset(
            [
                (*(data.numpy() for data in data), *(parameter.numpy() for parameter in parameters))
                for data, *parameters in zip(dataset, *target_distribution_parameters, strict=True)
            ]
        )
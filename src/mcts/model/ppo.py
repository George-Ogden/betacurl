from tensorflow.keras import losses
from tensorflow import data
import tensorflow as tf
import numpy as np

from typing import Callable, List, Optional, Tuple

from ...model import ModelFactory, TrainingConfig, BEST_MODEL_FACTORY
from ...game import GameSpec

from .config import PPOMCTSModelConfig

from .reinforce import MCTSModel

class PPOMCTSModel(MCTSModel):
    def __init__(
        self,
        game_spec: GameSpec,
        scaling_spec: Optional[np.ndarray] = None,
        model_factory: ModelFactory = BEST_MODEL_FACTORY,
        config: PPOMCTSModelConfig = PPOMCTSModelConfig()
    ):
        super().__init__(game_spec, scaling_spec, model_factory, config)
        self.target_kl = config.target_kl

    def compute_loss(
        self,
        observations: np.ndarray,
        action_groups: tf.RaggedTensor,
        values: tf.Tensor,
        advantage_groups: tf.RaggedTensor,
        target_distribution_params: tf.Tensor,
    ) -> tf.Tensor:
        predicted_distribution = self.generate_distribution(observations, training=True)
        predicted_values = self.predict_values(observations, training=True)
        target_distribution = type(predicted_distribution)(*tf.transpose(target_distribution_params, (1, 0, *range(2, target_distribution_params.ndim))))
        
        if isinstance(advantage_groups, tf.RaggedTensor) or isinstance(action_groups, tf.RaggedTensor):
            policy_loss = 0.
            approx_kl_div = 0.

            distribution_properties = {attr: getattr(predicted_distribution, attr) for attr in predicted_distribution.parameter_properties()}
            target_distribution_properties = {attr: getattr(target_distribution, attr) for attr in target_distribution.parameter_properties()}

            for i, (actions, advantages) in enumerate(zip(action_groups, advantage_groups, strict=True)):
                distribution = type(predicted_distribution)(**{k: v[i] for k, v in distribution_properties.items()})
                target_distribution = type(target_distribution)(**{k: v[i] for k, v in target_distribution_properties.items()})
                
                actions = actions.to_tensor()

                log_probs = tf.reduce_sum(distribution.log_prob(actions), axis=-1)
                target_log_probs = tf.reduce_sum(target_distribution.log_prob(actions), axis=-1)
                
                ratio = tf.exp(log_probs - target_log_probs)
                
                policy_loss_1 = ratio * advantages
                policy_loss_2 = tf.clip_by_value(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
                policy_loss -= tf.reduce_mean(tf.minimum(policy_loss_1, policy_loss_2))
                
                approx_kl_div += target_distribution.kl_divergence(distribution)
            policy_loss /= action_groups.shape[0]
            approx_kl_div /= action_groups.shape[0]
        else:
            action_groups = tf.transpose(action_groups, (1, 0, *range(2, action_groups.ndim)))
            advantage_groups = tf.transpose(advantage_groups, (1, 0, *range(2, advantage_groups.ndim)))
            log_probs = tf.reduce_sum(predicted_distribution.log_prob(action_groups), axis=-1)
            target_log_probs = tf.reduce_sum(target_distribution.log_prob(action_groups), axis=-1)

            ratio = tf.exp(log_probs - target_log_probs)

            policy_loss_1 = ratio * advantage_groups
            policy_loss_2 = tf.clip_by_value(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantage_groups
            policy_loss = -tf.reduce_mean(tf.minimum(policy_loss_1, policy_loss_2))

            approx_kl_div = target_distribution.kl_divergence(predicted_distribution)

        value_loss = losses.mean_squared_error(values, predicted_values)

        loss = policy_loss + self.vf_coeff * value_loss
        if self.ent_coeff != 0:
            # entropy is main cause of NaNs in training
            entropy_loss = -tf.reduce_mean(predicted_distribution.entropy())
            loss += self.ent_coeff * entropy_loss

        # stop early when KL divergence is too high
        if self.target_kl is not None and tf.reduce_mean(approx_kl_div) > 1.5 * self.target_kl:
            self.model.stop_training = True
            loss *= 0.

        return loss

    def preprocess_data(
        self,
        training_data: List[Tuple[int, np.ndarray, np.ndarray, float, List[Tuple[np.ndarray, float]]]],
        augmentation_function: Callable[[int, np.ndarray, np.ndarray, float], List[Tuple[int, np.ndarray, np.ndarray, float]]],
        training_config: TrainingConfig = TrainingConfig()
    ) -> data.Dataset:
        dataset = super().preprocess_data(training_data, augmentation_function, training_config)
        target_distributions = [
            self.generate_distribution(observation, training=False) 
            for observation, *_ in dataset.batch(training_config.batch_size)
        ]
        target_distribution_parameters = tf.concat([
            tf.stack([
                getattr(distribution, attr) 
                for attr in distribution.parameter_properties()
            ], axis=1)
            for distribution in target_distributions
        ], axis=0)
        return self.create_dataset(
            [
                (*[data.numpy() for data in data], parameters.numpy())
                for data, parameters in zip(dataset, target_distribution_parameters, strict=True)
            ]
        )
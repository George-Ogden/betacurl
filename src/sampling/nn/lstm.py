from tensorflow.keras import losses
import tensorflow as tf
import numpy as np

from dm_env.specs import BoundedArray
from typing import Optional, Tuple

from ...evaluation import NNEvaluationStrategy
from ...model import ModelFactory, ModelConfig

from .shared_torso import SharedTorsoSamplingEvaluatingStrategy

from tensorflow.keras import layers, losses
import tensorflow as tf
import numpy as np

from dm_env.specs import BoundedArray
from typing import Tuple

from ...evaluation import NNEvaluationStrategy
from ...model import DenseModelFactory, LSTMModelFactory, ModelConfig, ModelFactory, BEST_MODEL_FACTORY

from .gaussian import GaussianSamplingStrategy


class LSTMSamplingStrategy(SharedTorsoSamplingEvaluatingStrategy):
    def create_feature_extractor(self, input_size: int, feature_size: int, model_factory: Optional[ModelFactory] = None, config: Optional[ModelConfig] = None) -> tf.keras.Model:
        return LSTMModelFactory.create_model(input_size, feature_size, config)

    def compute_advantages_and_target_log_probs(
        self,
        players: tf.Tensor,
        observations: tf.Tensor,
        actions: tf.Tensor,
        reward: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        target_policy, target_values = self.target_model(
            self.preprocess_observations(
                observations
            ), training=False
        )
        target_distribution = self.generate_distribution(target_policy)

        advantages = reward - tf.squeeze(target_values, axis=-1)
        return advantages * tf.sign(players), self.compute_log_probs(target_distribution, actions)
    
    def compute_loss(self, observations: np.ndarray, actions: tf.Tensor, rewards: tf.Tensor, advantage: tf.Tensor, target_log_probs: tf.Tensor) -> tf.Tensor:
        predicted_policy, predicted_values = self.model(
            self.preprocess_observations(
                observations
            )
        )
        
        predicted_distribution = self.generate_distribution(predicted_policy)
        log_probs = self.compute_log_probs(predicted_distribution, actions)

        policy_loss = self.ppo_clip_loss(
            log_probs=log_probs,
            target_log_probs=target_log_probs,
            advantages=advantage
        )

        value_loss = losses.mean_squared_error(rewards, tf.squeeze(predicted_values, axis=-1))

        if self.target_kl is not None:
            log_ratio = log_probs - target_log_probs
            approx_kl_div = tf.reduce_mean((tf.exp(log_ratio) - 1) - log_ratio)

            if approx_kl_div > 1.5 * self.target_kl:
                self.model.stop_training = True

        return policy_loss + self.vf_coef * value_loss
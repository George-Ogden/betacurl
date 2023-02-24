from tensorflow.keras import losses
import tensorflow as tf
import numpy as np

from dm_env.specs import BoundedArray
from typing import Tuple

from ...evaluation import NNEvaluationStrategy
from ...model import DenseModelFactory, ModelFactory, BEST_MODEL_FACTORY

from ..config import SharedTorsoSamplerConfig
from .gaussian import GaussianSamplingStrategy

class SharedTorsoSamplingEvaluatingStrategy(GaussianSamplingStrategy, NNEvaluationStrategy):
    CONFIG_CLASS = SharedTorsoSamplerConfig
    DEFAULT_MODEL_FILE = "model.h5"
    def __init__(self, action_spec: BoundedArray, observation_spec: BoundedArray, model_factory: ModelFactory = BEST_MODEL_FACTORY, config: SharedTorsoSamplerConfig = SharedTorsoSamplerConfig()):
        super().__init__(action_spec=action_spec, observation_spec=observation_spec, config=config, model_factory=model_factory)
        # value function coefficient for loss calculation
        self.vf_coef = config.vf_coef
        self.target_kl = config.target_kl

    def setup_model(self, action_spec, observation_spec, model_factory, latent_size=0) -> tf.keras.Model:
        config = BEST_MODEL_FACTORY.CONFIG_CLASS(output_activation="linear")
        feature_size = self.config.feature_dim
        feature_extractor = model_factory.create_model(input_size=np.product(observation_spec.shape) + latent_size, output_size=feature_size, config=config)
        feature_spec = BoundedArray(shape=(feature_size,), dtype=np.float32, minimum=np.tile(-np.inf, feature_size), maximum=np.tile(np.inf, feature_size))
        policy_head = GaussianSamplingStrategy.setup_model(self, action_spec=action_spec, observation_spec=feature_spec, model_factory=DenseModelFactory)(feature_extractor.output)
        value_head = NNEvaluationStrategy.setup_model(self, observation_spec=feature_spec, model_factory=DenseModelFactory)(feature_extractor.output)
        self.model = tf.keras.Model(inputs=feature_extractor.inputs, outputs=[policy_head, value_head])
        return self.model

    def postprocess_actions(self, samples: tf.Tensor) -> tf.Tensor:
        actions, values = samples
        return super().postprocess_actions(actions)
    
    def postprocess_values(self, samples: tf.Tensor) -> tf.Tensor:
        actions, values = samples
        return super().postprocess_values(values)

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
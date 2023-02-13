from tensorflow.keras import losses
import tensorflow as tf
import numpy as np

from dm_env.specs import BoundedArray

from ...evaluation import NNEvaluationStrategy
from ...model import DenseModelFactory, ModelFactory, BEST_MODEL_FACTORY

from ..config import SharedTorsoSamplerConfig
from .gaussian import GaussianSamplingStrategy

class SharedTorsoSamplingStrategy(GaussianSamplingStrategy, NNEvaluationStrategy):
    CONFIG_CLASS = SharedTorsoSamplerConfig
    DEFAULT_MODEL_FILE = "model.h5"
    def __init__(self, action_spec: BoundedArray, observation_spec: BoundedArray, model_factory: ModelFactory = BEST_MODEL_FACTORY, config: SharedTorsoSamplerConfig = SharedTorsoSamplerConfig()):
        super().__init__(action_spec=action_spec, observation_spec=observation_spec, config=config, model_factory=model_factory)
        # value function coefficient for loss calculation
        self.vf_coef = config.vf_coefficient

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

    def compute_loss(self, observations: np.ndarray, actions: tf.Tensor, rewards: tf.Tensor) -> tf.Tensor:
        predicted_policy, predicted_values = self.model(
            self.preprocess_observations(
                observations
            )
        )
        
        target_policy, target_values = self.target_model(
            self.preprocess_observations(
                observations
            )
        )
        
        predicted_distribution = self.generate_distribution(predicted_policy)
        target_distribution = self.generate_distribution(target_policy)
        advantages = rewards - tf.squeeze(target_values, axis=-1)

        policy_loss = self.ppo_clip_loss(
            predicted_distribution=predicted_distribution,
            target_distribution=target_distribution,
            actions=actions,
            advantages=advantages
        )
        
        value_loss = losses.mean_squared_error(rewards, tf.squeeze(predicted_values, axis=-1))

        return value_loss * 0+  policy_loss
        return policy_loss + self.vf_coef * value_loss
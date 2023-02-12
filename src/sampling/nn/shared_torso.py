from tensorflow.keras import callbacks, layers
import tensorflow as tf
import numpy as np

from typing import Callable, List, Optional, Tuple
from dm_env.specs import BoundedArray

from ...evaluation import NNEvaluationStrategy
from ...model import DenseModelFactory, ModelDecorator, ModelFactory, TrainingConfig, BEST_MODEL_FACTORY

from ..config import SharedTorsoSamplerConfig
from .gaussian import GaussianSamplingStrategy

class SharedTorsoSamplingStrategy(NNEvaluationStrategy, GaussianSamplingStrategy, ModelDecorator):
    CONFIG_CLASS = SharedTorsoSamplerConfig
    DEFAULT_MODEL_FILE = "model.h5"
    def __init__(self, action_spec: BoundedArray, observation_spec: BoundedArray, model_factory: ModelFactory = BEST_MODEL_FACTORY, config: SharedTorsoSamplerConfig = SharedTorsoSamplerConfig()):
        super(GaussianSamplingStrategy, self).__init__(action_spec=action_spec, observation_spec=observation_spec, config=config, model_factory=model_factory)

    def setup_model(self, action_spec, observation_spec, model_factory, latent_size=0) -> tf.keras.Model:
        config = BEST_MODEL_FACTORY.CONFIG_CLASS(output_activation="linear")
        feature_size = self.config.feature_dim
        feature_extractor = model_factory.create_model(input_size=np.product(observation_spec.shape) + latent_size, output_size=feature_size, config=config)
        feature_spec = BoundedArray(shape=(feature_size,), dtype=np.float32, minimum=np.tile(-np.inf, feature_size), maximum=np.tile(np.inf, feature_size))
        policy_head = GaussianSamplingStrategy.setup_model(self, action_spec=action_spec, observation_spec=feature_spec, model_factory=DenseModelFactory)(feature_extractor.output)
        value_head = NNEvaluationStrategy.setup_model(self, observation_spec=feature_spec, model_factory=DenseModelFactory)(feature_extractor.output)
        self.model = tf.keras.Model(inputs=feature_extractor.inputs, outputs=[policy_head, value_head])
        return self.model

    def postprocess_actions(self, actions: tf.Tensor) -> tf.Tensor:
        actions = tf.reshape(actions, shape=(-1, *self.action_shape))
        actions *= self.action_range[1] - self.action_range[0]
        actions += self.action_range[0]
        return actions

    def normalise_outputs(self, actions: np.ndarray) -> np.ndarray:
        actions -= self.action_range[0]
        actions /= self.action_range[1] - self.action_range[0]
        return actions

    def generate_actions(self, observation: np.ndarray, n: Optional[int] = None) -> np.ndarray:
        batched_throughput = True
        if n is None:
            batched_throughput = False
            n = 1

        batch = np.tile(observation, (n, 1))
        batch = self.preprocess_observations(batch)
        input = self.add_noise_to_observations(batch)

        samples = self.model.predict(input, batch_size=256, verbose=0)
        samples = self.postprocess_actions(samples)

        if not batched_throughput:
            samples = tf.squeeze(samples, 0)

        samples = samples.numpy()
        return samples

    def learn(
        self,
        training_history: List[Tuple[int, np.ndarray, np.ndarray, float]],
        augmentation_function: Callable[[np.ndarray, np.ndarray, float], List[Tuple[np.ndarray, np.ndarray, float]]],
        training_config: TrainingConfig = TrainingConfig()
    ) -> callbacks.History:
        training_data = [(augmented_observation, augmented_action) for (player, observation, action, reward) in training_history for (augmented_observation, augmented_action, augmented_reward) in (augmentation_function(observation, action, reward) if np.sign(player) == np.sign(reward) else [])]
        observations, actions = zip(*training_data)

        observations = self.add_noise_to_observations(observations)

        return self.fit(observations, actions, training_config)
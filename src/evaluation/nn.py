from tensorflow.keras import callbacks
import tensorflow as tf
import numpy as np

from dm_env.specs import BoundedArray
from typing import Callable, List, Tuple

from ..model import ModelDecorator, ModelFactory, TrainingConfig, BEST_MODEL_FACTORY

from .base import EvaluationStrategy

class NNEvaluationStrategy(EvaluationStrategy, ModelDecorator):
    DEFAULT_MODEL_FILE = "evaluator.h5"
    def __init__(self, observation_spec: BoundedArray, model_factory: ModelFactory = BEST_MODEL_FACTORY):
        super().__init__(observation_spec)
        self.setup_model(observation_spec, model_factory)

    def setup_model(self, observation_spec: BoundedArray, model_factory: ModelFactory) -> tf.keras.Model:
        config = BEST_MODEL_FACTORY.CONFIG_CLASS(output_activation="linear")
        self.model: tf.keras.Model = model_factory.create_model(input_size=np.product(observation_spec.shape), output_size=1, config=config)
        return self.model
    
    def postprocess_values(self, values: tf.Tensor) -> tf.Tensor:
        return values

    def evaluate(self, observations: np.ndarray) -> np.ndarray:
        batched_throughput = False
        if observations.shape != self.observation_shape:
            assert observations.shape[1:] == self.observation_shape
            batched_throughput = True

        input = observations.reshape(-1, np.product(self.observation_shape))

        evaluations = self.model.predict(input, batch_size=256, verbose=0)
        evaluations = self.postprocess_values(evaluations)

        evaluations = tf.squeeze(evaluations, -1)
        if not batched_throughput:
            evaluations = tf.squeeze(evaluations, 0)
        return evaluations.numpy()

    def learn(
        self,
        training_history: List[Tuple[int, np.ndarray, np.ndarray, float]],
        augmentation_function: Callable[[np.ndarray, np.ndarray, float], List[Tuple[np.ndarray, np.ndarray, float]]],
        training_config: TrainingConfig = TrainingConfig()
    ) -> callbacks.History:
        training_data = [(augmented_observation, augmented_reward) for (player, observation, action, reward) in training_history for augmented_observation, augmented_action, augmented_reward in augmentation_function(observation, action, reward)]
        observations, values = zip(*training_data)
        return self.fit(observations, values, training_config)
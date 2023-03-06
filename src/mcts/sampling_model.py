from tensorflow.keras import callbacks, layers, losses
from tensorflow_probability import distributions
from tensorflow import keras
import tensorflow as tf
import numpy as np

from typing import Callable, List, Optional, Tuple, Union

from ..model import DenseModelFactory, ModelFactory, TrainingConfig, BEST_MODEL_FACTORY

from .model import MCTSModel
from .config import SamplingMCTSModelConfig

class SamplingMCTSModel(MCTSModel):
    MODELS = {
        "feature_extractor": "feature_extractor.h5",
        "policy_head": "policy.h5",
        "value_head": "value.h5",
        "observation_head": "observation.h5"
    }
    def __init__(
        self,
        game_spec: "GameSpec",
        scaling_spec: Optional[np.ndarray] = None,
        model_factory: ModelFactory = BEST_MODEL_FACTORY,
        config: SamplingMCTSModelConfig = SamplingMCTSModelConfig()
    ):
        super().__init__(
            game_spec=game_spec,
            scaling_spec=scaling_spec,
            model_factory=model_factory,
            config=config
        )

        self.num_samples = config.num_samples

    def _setup_model(self, model_factory: ModelFactory):
        super()._setup_model(model_factory)

        inputs = [
            keras.Input(self.feature_size),
            keras.Input(self.action_shape),
        ]

        self.observation_head = DenseModelFactory.create_model(
            input_shape=(self.feature_size + np.prod(self.action_shape)),
            output_shape=self.feature_size,
            config=DenseModelFactory.CONFIG_CLASS(
                output_activation="linear"
            )
        )

        self.observation_head = keras.Model(
            inputs=inputs,
            outputs=self.observation_head(
                layers.Concatenate()(
                    [   
                        inputs[0],
                        layers.Reshape(
                            (np.prod(self.action_shape, dtype=int),)
                        )(
                            inputs[1]
                        )
                    ]
                )
            )
        )

    def post_setup_model(self):
        inputs = [
            keras.Input(self.observation_shape),
            self.observation_head.inputs
        ]

        self.model = keras.Model(
            inputs=inputs,
            outputs=[
                self.policy_head(self.feature_extractor(inputs[0])),
                self.value_head(self.feature_extractor(inputs[0])),
                self.observation_head.outputs,
            ]
        )

        self.model([
            np.random.randn(1, *self.observation_shape),
            [
                np.random.randn(1, self.feature_size),
                np.random.randn(1, *self.action_shape)
            ]
        ])
    
    def predict_action_values(self, observation: Union[tf.Tensor, np.ndarray], actions: Union[tf.Tensor, np.ndarray], training: bool=False) -> Union[tf.Tensor, np.ndarray]:
        batch_throughput = True
        assert observation.shape == self.observation_shape
        if actions.ndim == len(self.action_shape):
            batch_throughput = False
            actions = np.expand_dims(actions, 0)

        observation = tf.expand_dims(observation, 0)

        feature = self.feature_extractor(observation, training=training)
        features = tf.tile(feature, (len(actions), 1))
        next_features = self.observation_head(
            [features, actions],
            training=training
        )

        values = self.value_head(next_features, training=training)

        if not batch_throughput:
            values = tf.squeeze(values, 0).numpy()

        return values

    def sample(self, distribution: distributions.Distribution) -> tf.Tensor:
        samples = distribution.sample(self.num_samples)
        predicted_values = self.predict_action_values(samples)
        return samples[tf.argmax(predicted_values)]

    def learn(
            self,
            training_history: List[Union[Tuple[int, np.ndarray, np.ndarray, float], Tuple[np.ndarray, np.ndarray, np.ndarray]]],
            augmentation_function: Callable[[int, np.ndarray, np.ndarray, float], List[Tuple[int, np.ndarray, np.ndarray, float]]],
            training_config: TrainingConfig = TrainingConfig(),
        ) -> callbacks.History:
        # interleaved history needs flattening
        training_history, transitions = (
            [move for history in training_history[::2] for move in history],
            [transition for history in training_history[1::2] for transition in history]
        )
        if len(transitions) > 0 and len(transitions[0]) != 3:
            training_history.append(transitions)

        history = super().learn(training_history, augmentation_function, training_config)
        
        if transitions is not None:
            # save model
            model = self.model
            # overwrite model to use APIs
            self.model = self.observation_head
            self.compile_model(training_config)

            # prepare dataset
            primary_dataset = self.create_dataset(transitions).batch(training_config.batch_size)

            # extract latent variables (these are frozen)
            latent_variables = [((self.feature_extractor(initial_observation), action), self.feature_extractor(final_observation)) for initial_observation, action, final_observation in primary_dataset]

            # fit the model with the old history
            X, Y, train_options = self.pre_fit(*zip(*latent_variables), training_config, convert_to_numpy=False)
            X0, X1, Y = [tf.concat(X, axis=0) for X in [*zip(*X), Y]]

            observation_history = self.model.fit([X0, X1], Y, **train_options)
            history = self.merge(history, observation_history)

            # restore model
            self.model = model
        
        return history

    def compute_loss(self, observations: np.ndarray, actions: tf.Tensor, values: tf.Tensor, advantages: tf.Tensor) -> tf.Tensor:
        return super().compute_loss(observations, actions, values, advantages)
from tensorflow_probability.python.internal import dtype_util
from tensorflow.keras import callbacks, layers, losses
from tensorflow_probability import distributions
from tensorflow import data, keras
import tensorflow as tf
import numpy as np

from dm_env.specs import BoundedArray
from copy import copy

from typing import Callable, List, Optional, Tuple, Union

from ...model import CustomDecorator, DenseModelFactory, ModelFactory, TrainingConfig, BEST_MODEL_FACTORY
from ...utils import SaveableMultiModel
from ...game import GameSpec

from .config import DiffusionMCTSModelConfig
from .base import MCTSModel

class DiffusionMCTSModel(MCTSModel):
    MODELS = {
        "feature_extractor": "feature_extractor.h5",
        "value_head": "value.h5",
        "action_encoder": "action_encoder.h5",
        "action_decoder": "action_decoder.h5",
        "diffusion_model": "diffusion.h5"
    }
    def __init__(
        self,
        game_spec: GameSpec,
        scaling_spec: Optional[np.ndarray] = None,
        model_factory: ModelFactory = BEST_MODEL_FACTORY,
        config: DiffusionMCTSModelConfig = DiffusionMCTSModelConfig()
    ):
        super().__init__(
            game_spec,
            scaling_spec=scaling_spec,
            config=config
        )
        self.diffusion_steps = config.diffusion_steps

        # precompute alphas and betas
        self.betas = np.exp(
            np.linspace(
                np.log(config.diffusion_coef_min),
                np.log(config.diffusion_coef_max),
                config.diffusion_steps,
                dtype=np.float32
            )
        )
        self.alphas = 1 - self.betas
        alphas_cumprod = np.cumprod(self.alphas)
        alphas_cumprod_prev = np.concatenate(([1], alphas_cumprod[:-1]), dtype=np.float32)
        self.noise_weight = np.sqrt(alphas_cumprod)
        self.image_weight = np.sqrt(1 - alphas_cumprod)
        self.posterior_variance = (
            self.betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        )

        self.feature_extractor = model_factory.create_model(
            input_shape=self.observation_shape,
            output_shape=self.feature_size,
            config=model_factory.CONFIG_CLASS(
                output_activation="linear"
            )
        )
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

        self.value_head = DenseModelFactory.create_model(
            input_shape=self.feature_size,
            output_shape=(),
            config=DenseModelFactory.CONFIG_CLASS(
                output_activation="linear"
            )
        )

        self.action_encoder = model_factory.create_model(
            input_shape=self.action_shape,
            output_shape=self.feature_size,
            config=model_factory.CONFIG_CLASS(
                output_activation="linear"
            )
        )

        self.action_decoder = model_factory.create_model(
            input_shape=self.feature_size * 2,
            output_shape=self.action_shape,
            config=model_factory.CONFIG_CLASS(
                output_activation="sigmoid"
            )
        )
        # TODO: use scaling spec
        self.action_decoder = keras.Sequential([
            self.action_decoder,
            layers.Rescaling(
                scale=self.action_range[1] - self.action_range[0],
                offset=self.action_range[0]
            )
        ])

        self.setup_model()
    
    def setup_model(self):
        observation_input = layers.Input(shape=self.observation_shape)
        action_input = layers.Input(shape=self.action_shape)
        features = self.feature_extractor(observation_input)
        self.diffusion_model = keras.Model(
            inputs=[action_input, observation_input],
            outputs=self.action_decoder(
                layers.Concatenate()([
                    self.action_encoder(action_input),
                    features
                ])
            )
        )
        self.model = keras.Model(
            inputs=[observation_input, action_input],
            outputs=[
                self.value_head(features),
                self.diffusion_model.outputs
            ]
        )

    def generate_distribution(
        self,
        observation: Union[tf.Tensor, np.ndarray],
        training: bool = False
    ) -> distributions.Distribution:
        return DiffusionDistribution(
            model=self,
            observation=observation,
            training=training
        )

    def generate_samples(
        self,
        observation: Union[tf.Tensor, np.ndarray],
        training: bool = False
    ) -> tf.Tensor:
        actions = tf.random.normal(shape=(observation.shape[0],) + self.action_shape, dtype=tf.float32)
        for i in reversed(range(self.diffusion_steps)):
            actions += tf.random.normal(shape=(actions.shape), dtype=tf.float32) * np.sqrt(self.posterior_variance[i])
            actions = self.diffusion_model([actions, observation], training=training)
        return actions
    
    def predict_values(self, observation: Union[tf.Tensor, np.ndarray], training: bool = False) -> Union[tf.Tensor, np.ndarray]:
        ...
    
    def compute_loss(self, *batch: List[tf.Tensor]) -> tf.Tensor:
        ...

class DiffusionDistribution(distributions.Distribution):
    batch_size: int = 1024
    def __init__(
        self,
        model: DiffusionMCTSModel,
        observation: tf.Tensor,
        training: bool = False,
        validate_args: bool = False,
    ):
        dtype = dtype_util.common_dtype([observation], dtype_hint=tf.float32)
        super().__init__(
            dtype=dtype,
            reparameterization_type=distributions.FULLY_REPARAMETERIZED,
            validate_args=validate_args,
            allow_nan_stats=True,
            name="DiffusionDistribution"
        )
        self.model = model
        self.observation = observation
        self.training = training
        self.samples = tf.zeros(shape=(0, *self.model.action_shape), dtype=tf.float32)
        self.sampled = 0

    def _sample_n(self, n: int, seed=None):
        self.make_samples(n + self.sampled)
        self.sampled += n
        return self.samples[self.sampled - n:self.sampled]

    def make_samples(self, n: int):
        while n > len(self.samples):
            self.samples = tf.concat(
                (
                    self.samples,
                    self.model.generate_samples(
                        tf.reshape(
                            tf.tile(self.observation, (self.batch_size,) +  (1,) * (self.observation.ndim - 1)),
                            (self.batch_size, *self.observation.shape)
                        ),
                    )
                ),
                axis=0
            )

    def _mean(self):
        self.make_samples(1)
        return tf.reduce_mean(self.samples, axis=0)

    def _stddev(self):
        self.make_samples(1)
        return tf.math.reduce_std(self.samples, axis=0)

    def prob(self, action: tf.Tensor):
        return tf.ones_like(action)
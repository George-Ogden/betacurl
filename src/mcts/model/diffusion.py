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
    
    def _generate_distribution(self, raw_actions: tf.Tensor) -> distributions.Distribution:
        ...
    
    def predict_values(self, observation: Union[tf.Tensor, np.ndarray], training: bool = False) -> Union[tf.Tensor, np.ndarray]:
        ...
    
    def compute_loss(self, *batch: List[tf.Tensor]) -> tf.Tensor:
        ...
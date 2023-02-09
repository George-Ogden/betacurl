from .config import ModelConfig, MLPModelConfig, TrainingConfig
from .mlp import MLPModelFactory
from .decorator import Learnable, ModelDecorator
from .fcnn import MultiLayerModelFactory
from .base import ModelFactory

BEST_MODEL_FACTORY = MultiLayerModelFactory()
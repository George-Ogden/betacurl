from .decorator import CustomDecorator, Learnable, ModelDecorator
from .config import ModelConfig, MLPModelConfig, TrainingConfig
from .fcnn import MultiLayerModelFactory
from .dense import DenseModelFactory
from .mlp import MLPModelFactory
from .base import ModelFactory

BEST_MODEL_FACTORY = MultiLayerModelFactory()
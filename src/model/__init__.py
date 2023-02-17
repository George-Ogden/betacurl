from .config import ModelConfig, MLPModelConfig, TrainingConfig
from .decorator import Learnable, ModelDecorator
from .fcnn import MultiLayerModelFactory
from .dense import DenseModelFactory
from .lstm import LSTMModelFactory
from .mlp import MLPModelFactory
from .base import ModelFactory

BEST_MODEL_FACTORY = MultiLayerModelFactory()
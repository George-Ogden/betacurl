from .config import ModelConfig, SimpleLinearModelConfig, TrainingConfig
from .simple_linear import SimpleLinearModelFactory
from .decorator import Learnable, ModelDecorator
from .fcnn import FCNNModelFactory
from .base import ModelFactory

BEST_MODEL_FACTORY = FCNNModelFactory
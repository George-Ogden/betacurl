from .config import ModelConfig, SimpleLinearModelConfig, TrainingConfig
from .simple_linear import SimpleLinearModelFactory
from .decorator import ModelDecorator, Learnable
from .fcnn import FCNNModelFactory
from .base import ModelFactory

BEST_MODEL_FACTORY = FCNNModelFactory
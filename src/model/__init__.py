from .config import ModelConfig, SimpleLinearModelConfig
from .simple_linear import SimpleLinearModelFactory
from .fcnn import FCNNModelFactory
from .base import ModelFactory

BEST_MODEL_FACTORY = FCNNModelFactory
from src.model.config import ModelConfig, SimpleLinearModelConfig
from src.model.base import ModelFactory
from src.model.simple_linear import SimpleLinearModelFactory
from src.model.fcnn import FCNNModelFactory

BEST_MODEL_FACTORY = FCNNModelFactory
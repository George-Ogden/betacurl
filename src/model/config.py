from dataclasses import dataclass

@dataclass
class ModelConfig:
    output_activation:  str = "sigmoid"

@dataclass
class SimpleLinearModelConfig(ModelConfig):
    hidden_size: int = 128

@dataclass
class FCNNConfig(SimpleLinearModelConfig):
    dropout: float = .1
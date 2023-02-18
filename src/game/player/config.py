from dataclasses import dataclass, field
from typing import Union

from ...sampling import SamplerConfig

@dataclass
class SamplingEvaluatingPlayerConfig:
    sampler_config: Union[SamplerConfig, dict] = field(default_factory=dict)
    num_train_samples: int = 100
    """number of samples generated during training"""
    num_eval_samples: int = 100
    """number of samples generated during evaluation"""
    epsilon: float = 0.1
    """epsilon-greedy exploration parameter"""
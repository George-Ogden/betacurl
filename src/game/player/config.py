from dataclasses import dataclass
from typing import Optional

@dataclass
class SamplingEvaluatingPlayerConfig:
    num_train_samples: int = 100
    """number of samples generated during training"""
    num_eval_samples: int = 100
    """number of samples generated during evaluation"""
    epsilon: float = 0.1
    """epsilon-greedy exploration parameter"""
    latent_size: Optional[int] = None
    """size of latent space for sample generation"""
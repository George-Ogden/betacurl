import numpy as np

from dataclasses import dataclass, field
from typing import Optional, Union

from ...sampling import SamplerConfig
from ...mcts import MCTSConfig
from ...utils import Config

@dataclass
class SamplingEvaluatingPlayerConfig(Config):
    sampler_config: Union[SamplerConfig, dict] = field(default_factory=dict)
    num_train_samples: int = 100
    """number of samples generated during training"""
    num_eval_samples: int = 100
    """number of samples generated during evaluation"""
    epsilon: float = 0.1
    """epsilon-greedy exploration parameter"""

@dataclass
class MCTSPlayerConfig(Config):
    mcts_config: Union[MCTSConfig, dict] = field(default_factory=dict)
    num_simulations: int = 50
    """number of fresh moves to simulate in each position"""

    def __post_init__(self):
        # one simulation doesn't generate any actions
        assert self.num_simulations >= 2

@dataclass
class NNMCTSPlayerConfig(MCTSPlayerConfig):
    scaling_spec: Optional[np.ndarray] = None
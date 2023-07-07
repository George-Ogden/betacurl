from dataclasses import dataclass
from typing import Optional

from ...distribution import DistributionConfig
from ...utils import Config

@dataclass
class MCTSModelConfig(Config):
    feature_size: int = 128
    vf_coeff: float = .5
    max_grad_norm: float = .5    
    distribution_config: Optional[DistributionConfig] = None

@dataclass
class PolicyMCTSModelConfig(MCTSModelConfig):
    ent_coeff: float = 1e-3

@dataclass
class PPOMCTSModelConfig(PolicyMCTSModelConfig):
    clip_range: float = .1
    target_kl: Optional[float] = 1.5
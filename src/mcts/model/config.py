from typing import ClassVar, Optional
from dataclasses import dataclass

from ...utils import Config
from ...distribution import DistributionConfig

@dataclass
class MCTSModelConfig(Config):
    feature_size: int = 64
    vf_coeff: float = .5
    max_grad_norm: float = .5    

@dataclass
class PolicyMCTSModelConfig(MCTSModelConfig):
    ent_coeff: float = 0.01
    distribution_config: Optional[DistributionConfig] = None

@dataclass
class ReinforceMCTSModelConfig(PolicyMCTSModelConfig):
    clip_range: float = 2.

@dataclass
class PPOMCTSModelConfig(ReinforceMCTSModelConfig):
    clip_range: float = .1
    target_kl: Optional[float] = 1.5
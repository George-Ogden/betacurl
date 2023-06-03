from typing import ClassVar, Optional
from dataclasses import dataclass

from ...utils import Config

@dataclass
class MCTSModelConfig(Config):
    feature_size: int = 64
    vf_coeff: float = .5
    max_grad_norm: float = .5

@dataclass
class PolicyMCTSModelConfig(MCTSModelConfig):
    ent_coeff: float = 0.
    exploration_coefficient: float = 1.
    distribution_granularity: int = 64

@dataclass
class ReinforceMCTSModelConfig(PolicyMCTSModelConfig):
    clip_range: float = 2.
    exploration_coefficient: ClassVar[float] = 0.

@dataclass
class PPOMCTSModelConfig(ReinforceMCTSModelConfig):
    distribution_granularity: ClassVar[int] = 0
    clip_range: float = .1
    target_kl: Optional[float] = 1.5
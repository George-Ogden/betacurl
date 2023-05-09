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
    fourier_features: int = 3

@dataclass
class ReinforceMCTSModelConfig(PolicyMCTSModelConfig):
    clip_range: float = 2.

@dataclass
class PPOMCTSModelConfig(ReinforceMCTSModelConfig):
    fourier_features: ClassVar[int] = 0
    clip_range: float = .1
    target_kl: Optional[float] = 1.5
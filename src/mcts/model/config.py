from dataclasses import dataclass
from typing import Optional

from ...utils import Config

@dataclass
class MCTSModelConfig(Config):
    feature_size: int = 32
    vf_coeff: float = .5
    max_grad_norm: float = .5

@dataclass
class ReinforceMCTSModelConfig(MCTSModelConfig):
    ent_coeff: float = 0.
    clip_range: float = 2.

@dataclass
class PPOMCTSModelConfig(ReinforceMCTSModelConfig):
    target_kl: Optional[float] = 1.5
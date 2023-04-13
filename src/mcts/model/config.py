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
    clip_range: float = .1
    target_kl: Optional[float] = 1.5

@dataclass
class DiffusionMCTSModelConfig(MCTSModelConfig):
    diffusion_coef_min: float = 2e-4
    diffusion_coef_max: float = 2e-2
    diffusion_steps: int = 10

@dataclass
class FourierMCTSModelConfig(PPOMCTSModelConfig):
    fourier_features: int = 4
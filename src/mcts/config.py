from dataclasses import dataclass

from ..utils.config import Config

@dataclass
class MCTSConfig(Config):
    cpuct: float = 1. # "theoretically equal to √2; in practice usually chosen empirically"

@dataclass
class FixedMCTSConfig(MCTSConfig):
    num_actions: int = 10

@dataclass
class WideningMCTSConfig(MCTSConfig):
    # m(s) = c_pw * n(s)^kappa
    cpw: float = 1.
    kappa: float = .5
    def __post_init__(self):
        assert 0 < self.kappa and self.kappa <= 1

@dataclass
class MCTSModelConfig(Config):
    feature_size: int = 32
    vf_coeff: float = 10.
    ent_coeff: float = 1e-3
    max_grad_norm: float = .5
    clip_range: float = 2.

@dataclass
class SamplingMCTSModelConfig(MCTSModelConfig):
    ...

@dataclass
class NNMCTSConfig(WideningMCTSConfig):
    max_rollout_depth: int = 4
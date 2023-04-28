from dataclasses import dataclass

from ..utils.config import Config

@dataclass
class MCTSConfig(Config):
    cpuct: float = 1. # "theoretically equal to √2; in practice usually chosen empirically"
    scale_reward: bool = False

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
class WideningNNMCTSConfig(WideningMCTSConfig):
    max_rollout_depth: int = 4
    """maximum depth to run rollout"""

@dataclass
class FixedNNMCTSConfig(FixedMCTSConfig):
    max_rollout_depth: int = 4
    """maximum depth to run rollout"""
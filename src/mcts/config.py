from dataclasses import dataclass

@dataclass
class MCTSConfig:
    cpuct: float = 1. # "theoretically equal to √2; in practice usually chosen empirically"

@dataclass
class FixedMCTSConfig(MCTSConfig):
    num_actions: int = 10

@dataclass
class WideningMCTSConfig(MCTSConfig):
    # m(s) = c_pw * n(s)^kappa
    kappa: float = .5
    cpw: float = 1.
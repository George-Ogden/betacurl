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
    cpw: float = 1.
    kappa: float = .5
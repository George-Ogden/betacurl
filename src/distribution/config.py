from dataclasses import dataclass

from ..utils import Config

@dataclass
class DistributionConfig(Config):
    ...

@dataclass
class CombDistributionConfig(DistributionConfig):
    granularity: int = 64
    noise_ratio: float = .05

class NormalDistributionConfig(DistributionConfig):
    ...
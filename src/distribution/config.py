from dataclasses import dataclass

from ..utils import Config

@dataclass
class DistributionConfig(Config):
    noise_ratio: float = .05

@dataclass
class CombDistributionConfig(DistributionConfig):
    granularity: int = 64

class NormalDistributionConfig(DistributionConfig):
    ...
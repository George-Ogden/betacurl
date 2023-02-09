from dataclasses import dataclass
from typing import ClassVar, List
import typing


@dataclass
class SamplerConfig:
    def keys(self) -> List[str]:
        return list(filter(lambda key: typing.get_origin(self.__annotations__[key]) is not ClassVar, self.__dataclass_fields__.keys()))

    def __getitem__(self, key):
        return getattr(self, key)

@dataclass
class NNSamplerConfig(SamplerConfig):
    latent_size: int = 4
    """size of additional noise vector used to produce variation"""

@dataclass
class GaussianSamplerConfig(NNSamplerConfig):
    latent_size: ClassVar[int] = 0
    clip_ratio: float = .1
    """PPO clip ratio"""
    max_grad_norm: float = .5

@dataclass
class SharedTorsoSamplerConfig(GaussianSamplerConfig):
    feature_dim: int = 32
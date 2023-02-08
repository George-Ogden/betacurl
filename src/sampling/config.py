from dataclasses import dataclass
from typing import ClassVar, Iterable


@dataclass
class SamplerConfig:
    def keys(self) -> Iterable[str]:
        return self.__dataclass_fields__.keys()

    def __getitem__(self, key):
        return getattr(self, key)

@dataclass
class NNSamplerConfig(SamplerConfig):
    latent_size: int = 4

@dataclass
class GaussianNNSamplerConfig(NNSamplerConfig):
    latent_size: ClassVar[int] = 0
    clip_ratio: float = .1
    target_update_frequency: int = 2
    max_grad_norm: float = .5
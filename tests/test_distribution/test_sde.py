from typing import Type

from src.distribution import DistributionFactory, NormalSDEDistributionFactory
from src.distribution.config import SDEDistributionConfig

import pytest

SDEDistributions = [
    NormalSDEDistributionFactory
]

@pytest.fixture(params=SDEDistributions)
def Factory(request):
    return request.param

def test_distribution_config(Factory: Type[DistributionFactory]):
    config = Factory.CONFIG_CLASS()
    assert config.noise_ratio == SDEDistributionConfig.noise_ratio
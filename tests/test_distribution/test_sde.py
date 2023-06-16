import tensorflow as tf
import numpy as np

from typing import Type

import pytest

from src.distribution import DistributionFactory, NormalSDEDistributionFactory
from src.distribution.config import SDEDistributionConfig

from tests.utils import MDPStubGame

game = MDPStubGame(10)
move_spec = game.game_spec.move_spec

SDEDistributions = [
    NormalSDEDistributionFactory
]

@pytest.fixture(params=SDEDistributions)
def Factory(request):
    return request.param

def test_distribution_config(Factory: Type[DistributionFactory]):
    config = Factory.CONFIG_CLASS()
    assert config.noise_ratio == SDEDistributionConfig.noise_ratio

def test_noise_continuation(Factory: Type[DistributionFactory]):
    config: SDEDistributionConfig = Factory.CONFIG_CLASS(
        exploration_steps=2
    )
    distribution_factory = Factory(
        move_spec=move_spec,
        config=config
    )
    distribution_factory.noise_on()
    parameters = tf.random.normal((4, 2) + distribution_factory.parameters_shape)
    features = tf.random.normal((4, 2, 32))
    distribution_1 = distribution_factory.create_distribution(parameters, features=features)
    distribution_2 = distribution_factory.create_distribution(parameters, features=features)
    distribution_3 = distribution_factory.create_distribution(parameters, features=features)
    assert np.allclose(distribution_1.kl_divergence(distribution_2), 0.)
    assert np.any(distribution_1.kl_divergence(distribution_3) > 0)
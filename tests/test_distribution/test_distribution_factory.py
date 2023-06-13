import tensorflow as tf
import numpy as np

from typing import Type

import pytest

from src.distribution import DistributionFactory, DistributionConfig, CombDistributionFactory
from src.distribution.comb import CombDistribution

from tests.utils import MDPStubGame

game = MDPStubGame(10)
move_spec = game.game_spec.move_spec

distribution_mapping = {
    CombDistributionFactory: CombDistribution
}
@pytest.fixture(params=list(distribution_mapping))
def DistributionFactory(request):
    return request.param

default_config = DistributionConfig()

def test_distribution_initialsation(DistributionFactory: Type[DistributionFactory]):
    config = DistributionFactory.CONFIG_CLASS(
        **default_config
    )
    assert not type(config) is type(default_config)
    distribution_factory = DistributionFactory(
        move_spec=move_spec,
        config=config
    )
    assert isinstance(distribution_factory, DistributionFactory)

def test_distribution_parameters(DistributionFactory: Type[DistributionFactory]):
    distribution_factory = DistributionFactory(
        move_spec=move_spec
    )
    assert isinstance(distribution_factory.parameters_shape, tuple)
    for shape in distribution_factory.parameters_shape:
        assert isinstance(shape, int)
        assert shape > 0

def test_distribution_generation(DistributionFactory: Type[DistributionFactory]):
    Distribution = distribution_mapping[DistributionFactory]
    distribution_factory = DistributionFactory(
        move_spec=move_spec
    )
    parameter_shape = distribution_factory.parameters_shape
    random_parameters = tf.random.normal((4, 2) + parameter_shape)
    distribution = distribution_factory.create_distribution(random_parameters)
    assert isinstance(distribution, Distribution)
    assert distribution.sample().shape[0:2] == (4, 2)
    for _ in range(100):
        for sample in distribution.sample().numpy().reshape((-1,) + move_spec.shape):
            move_spec.validate(sample)

def test_distribution_noise_switching(DistributionFactory: Type[DistributionFactory]):
    distribution_factory = DistributionFactory(
        move_spec=move_spec
    )
    distribution_factory.noise_off()
    parameters = tf.random.normal(distribution_factory.parameters_shape)
    distribution1 = distribution_factory.create_distribution(parameters)
    distribution2 = distribution_factory.create_distribution(parameters)
    assert np.allclose(distribution1.kl_divergence(distribution2), 0.)
    
    distribution_factory.noise_on()
    distribution3 = distribution_factory.create_distribution(parameters)
    distribution4 = distribution_factory.create_distribution(parameters)
    assert np.any(distribution3.kl_divergence(distribution4) > 0.)

def test_parameterisation(DistributionFactory: Type[DistributionFactory]):
    distribution_factory = DistributionFactory(
        move_spec=move_spec
    )
    distribution_factory.noise_off()
    parameters = tf.random.normal(distribution_factory.parameters_shape)
    original_distribution = distribution_factory.create_distribution(parameters)
    
    sample = original_distribution.sample()
    parameters = distribution_factory.parameterize(sample)
    
    new_distribution = distribution_factory.create_distribution(parameters)
    assert new_distribution.sample().shape == sample.shape
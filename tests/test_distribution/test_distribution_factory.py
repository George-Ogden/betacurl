from tensorflow_probability import distributions
import tensorflow as tf
import numpy as np

from typing import Type

import pytest

from src.distribution import DistributionFactory, DistributionConfig, CombDistributionFactory, NormalDistributionFactory, NormalSDEDistributionFactory
from src.distribution.comb import CombDistribution

from tests.utils import MDPStubGame

game = MDPStubGame(10)
move_spec = game.game_spec.move_spec

distribution_mapping = {
    CombDistributionFactory: CombDistribution,
    NormalDistributionFactory: distributions.Normal,
    NormalSDEDistributionFactory: distributions.Normal
}
@pytest.fixture(params=list(distribution_mapping))
def Factory(request):
    return request.param

default_config = DistributionConfig()

def test_distribution_initialsation(Factory: Type[DistributionFactory]):
    config = Factory.CONFIG_CLASS(
        **default_config
    )
    assert not type(config) is type(default_config)
    distribution_factory = Factory(
        move_spec=move_spec,
        config=config
    )
    assert isinstance(distribution_factory, DistributionFactory)

def test_distribution_parameters(Factory: Type[DistributionFactory]):
    distribution_factory = Factory(
        move_spec=move_spec
    )
    assert isinstance(distribution_factory.parameters_shape, tuple)
    for shape in distribution_factory.parameters_shape:
        assert isinstance(shape, int)
        assert shape > 0

def test_distribution_generation(Factory: Type[DistributionFactory]):
    Distribution = distribution_mapping[Factory]
    distribution_factory = Factory(
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

def test_distribution_noise_switching(Factory: Type[DistributionFactory]):
    distribution_factory = Factory(
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

def test_parameterisation(Factory: Type[DistributionFactory]):
    distribution_factory = Factory(
        move_spec=move_spec
    )
    distribution_factory.noise_off()
    parameters = tf.random.normal(distribution_factory.parameters_shape)
    original_distribution = distribution_factory.create_distribution(parameters)
    
    sample = original_distribution.sample()
    parameters = distribution_factory.parameterize(sample)
    
    new_distribution = distribution_factory.create_distribution(parameters)
    assert new_distribution.sample().shape == sample.shape

def test_aggregation(Factory: Type[DistributionFactory]):
    distribution_factory = Factory(
        move_spec=move_spec
    )
    parameters = [
        (
            tf.random.normal(distribution_factory.parameters_shape),
            np.random.randint(1, 10)
        )
        for _ in range(10)
    ]
    aggregated_parameters = distribution_factory.aggregate_parameters(parameters)
    assert aggregated_parameters.shape == distribution_factory.parameters_shape
    distribution = distribution_factory.create_distribution(aggregated_parameters)
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
    random_features = tf.random.normal((4, 2, 32))
    distribution = distribution_factory.create_distribution(random_parameters, features=random_features)
    assert isinstance(distribution, Distribution)
    assert distribution.sample().shape[0:2] == (4, 2)
    for _ in range(100):
        for sample in distribution.sample().numpy().reshape((-1,) + move_spec.shape):
            move_spec.validate(sample)

def test_distribution_noise_switching(Factory: Type[DistributionFactory]):
    distribution_factory = Factory(
        move_spec=move_spec,
        config=Factory.CONFIG_CLASS(
            noise_ratio=1.
        )
    )
    distribution_factory.noise_off()
    parameters = tf.random.normal(distribution_factory.parameters_shape)
    features = tf.random.normal((32,))
    distribution_1 = distribution_factory.create_distribution(parameters, features=features)
    distribution_2 = distribution_factory.create_distribution(parameters, features=features)
    assert np.allclose(distribution_1.kl_divergence(distribution_2), 0.)
    
    distribution_factory.noise_on()
    distribution_3 = distribution_factory.create_distribution(parameters, features=features)
    distribution_factory.noise_on() # SDE requires resetting noise for difference 
    distribution_4 = distribution_factory.create_distribution(parameters, features=features)
    assert np.any(distribution_3.kl_divergence(distribution_4) > 0.)

def test_parameterisation(Factory: Type[DistributionFactory]):
    distribution_factory = Factory(
        move_spec=move_spec
    )
    distribution_factory.noise_off()
    parameters = tf.random.normal(distribution_factory.parameters_shape)
    features = tf.random.normal((32,))
    original_distribution = distribution_factory.create_distribution(parameters, features=features)
    
    sample = original_distribution.sample()
    parameters = distribution_factory.parameterize(sample)
    
    new_distribution = distribution_factory.create_distribution(parameters, features=features)
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
    distribution = distribution_factory.create_distribution(aggregated_parameters, features=tf.random.normal((32,)))
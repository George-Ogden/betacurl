import tensorflow as tf

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
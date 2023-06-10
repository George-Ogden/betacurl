from typing import Type

from pytest import mark

from src.distribution import DistributionFactory, DistributionConfig, CombDistributionFactory

from tests.utils import MDPStubGame

game = MDPStubGame(10)
move_spec = game.game_spec.move_spec

multiple_distributions = mark.parametrize(
    "DistributionFactory", [
        CombDistributionFactory,
    ]
)

default_config = DistributionConfig()

@multiple_distributions
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

@multiple_distributions
def test_distribution_parameters(DistributionFactory: Type[DistributionFactory]):
    distribution_factory = DistributionFactory(
        move_spec=move_spec
    )
    assert isinstance(distribution_factory.parameters_shape, tuple)
    for shape in distribution_factory.parameters_shape:
        assert isinstance(shape, int)
        assert shape > 0
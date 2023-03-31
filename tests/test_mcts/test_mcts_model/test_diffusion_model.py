import tensorflow as tf
import numpy as np

from dm_env.specs import Array, BoundedArray
from pytest import mark

from src.mcts import DiffusionMCTSModel, DiffusionMCTSModelConfig
from src.game import GameSpec

from tests.utils import MDPStubGame

game = MDPStubGame(6)
game_spec = game.game_spec
observation_spec = game_spec.observation_spec
move_spec = game_spec.move_spec

model = DiffusionMCTSModel(game_spec=game_spec)

def test_config_is_used():
    model = DiffusionMCTSModel(
        game_spec=game_spec,
        config=DiffusionMCTSModelConfig(
            diffusion_coef_max=1e-1,
            diffusion_coef_min=1e-5,
            diffusion_steps = 9
        )
    )

    assert (np.argsort(model.betas) == np.arange(9, dtype=int)).all()
    assert np.allclose(model.betas[0], 1e-5)
    assert np.allclose(model.betas[-1], 1e-1)
    for array in (
        model.betas,
        model.noise_weight,
        model.action_weight,
        model.alphas,
        model.posterior_variance,
    ):
        assert np.all(array >= 0) and np.all(array <= 1)
        assert array.dtype == np.float32

def test_distribution():
    observation = np.random.uniform(low=observation_spec.minimum, high=observation_spec.maximum)
    distribution = model.generate_distribution(observation)
    mean = distribution.mean().numpy()
    assert mean.shape == move_spec.shape
    assert (mean >= move_spec.minimum).all() and (mean <= move_spec.maximum).all()
    
    std = distribution.stddev().numpy()
    assert std.shape == move_spec.shape
    assert (std > 0.).all() and (std < np.sqrt(move_spec.maximum - move_spec.minimum)).all()
    
    for _ in range(10000):
        sample = distribution.sample().numpy()
        assert sample.shape == move_spec.shape
        assert (sample >= move_spec.minimum).all() and (sample <= move_spec.maximum).all()
        assert np.prod(distribution.prob(sample).numpy()) > 0.

@mark.flaky
def test_half_specified_scaling_spec():
    game_spec = GameSpec(
        move_spec=BoundedArray(
            shape=(3,),
            dtype=np.float32,
            minimum=np.array([0., -1, 0.]),
            maximum=np.array([2., 1., 2.]),
        ),
        observation_spec=Array(
            shape=(3,),
            dtype=np.float32,
        )
    )
    model = DiffusionMCTSModel(
        game_spec=game_spec,
        scaling_spec=np.array([.5, .5, .5])
    )
    output = model.action_decoder(
        np.random.randn(1000, model.action_decoder.input_shape[-1])
    )
    mean = tf.reduce_mean(output)
    std = tf.math.reduce_std(output)
    assert np.allclose(mean, .5, atol=.5)
    assert np.allclose(std, 1., atol=.5)

@mark.flaky
def test_fully_specified_scaling_spec():
    game_spec = GameSpec(
        move_spec=BoundedArray(
            shape=(3,),
            dtype=np.float32,
            minimum=np.array([0., -2, 0.]),
            maximum=np.array([4., 2., 4.]),
        ),
        observation_spec=Array(
            shape=(3,),
            dtype=np.float32,
        )
    )
    model = DiffusionMCTSModel(
        game_spec=game_spec,
        scaling_spec=np.array([[1., .5],[1., 1.],[1., 2.]])
    )
    output = model.action_decoder(
        np.random.randn(1000, model.action_decoder.input_shape[-1])
    )
    mean = tf.reduce_mean(output)
    std = tf.math.reduce_std(output, axis=0)
    assert np.allclose(mean, 1., atol=1.)
    assert tf.reduce_all([0., .5, 1.] < std) and tf.reduce_all(std < [1., 1.5, 3.])

from tensorflow_probability import distributions
from tensorflow.keras import callbacks
from dm_env.specs import BoundedArray
from copy import deepcopy
import tensorflow as tf
import numpy as np

from src.model import SimpleLinearModelFactory, TrainingConfig
from src.game import Game

from src.sampling import GaussianSamplingStrategy

from tests.utils import EpochCounter
from tests.config import probabilistic

Normal = distributions.Normal

single_action_spec = BoundedArray(minimum=(-100,), maximum=(100,), shape=(1,), dtype=np.float32)
single_observation_spec = BoundedArray(minimum=(-100,), maximum=(100,), shape=(1,), dtype=np.float32)
single_action_strategy = GaussianSamplingStrategy(action_spec=single_action_spec, observation_spec=single_observation_spec, model_factory=SimpleLinearModelFactory)

wide_action_spec = BoundedArray(minimum=(-100,-100), maximum=(100, 100), shape=(2,), dtype=np.float32)
wide_observation_spec = BoundedArray(minimum=(-100,-100), maximum=(100, 100), shape=(2,), dtype=np.float32)
wide_range_strategy = GaussianSamplingStrategy(action_spec=wide_action_spec, observation_spec=wide_observation_spec, model_factory=SimpleLinearModelFactory)

skewed_action_spec = BoundedArray(minimum=(0.,), maximum=(200.,), shape=(1,), dtype=np.float32)
narrow_skewed_action_spec = BoundedArray(minimum=(99.9,), maximum=(100.,), shape=(1,), dtype=np.float32)

skewed_strategy = GaussianSamplingStrategy(action_spec=skewed_action_spec, observation_spec=wide_observation_spec)
narrow_skewed_strategy = GaussianSamplingStrategy(action_spec=narrow_skewed_action_spec, observation_spec=wide_observation_spec)

@probabilistic
def test_samples_are_normal():
    means = np.ones(1000)
    log_stds = np.log(np.ones(1000) * 2)
    actions = np.concatenate((means, log_stds), axis=-1).reshape(-1, 4)
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        samples = wide_range_strategy.postprocess_actions(actions)
        mean = float(tf.reduce_mean(samples).numpy())
        std = float(tf.math.reduce_std(samples).numpy())
    assert .5 < mean < 1.5
    assert 1. < std < 3.

def test_distribution_is_correct():
    distribution = wide_range_strategy.generate_distribution(np.array(((-1., 2., -2, 3.),)))
    assert np.allclose(distribution.mean(), (-1, -2)) and np.allclose(distribution.stddev(), np.exp((2, 3)))

@probabilistic
def test_action_skew():
    actions = skewed_strategy.generate_actions(np.random.randn(2), n=100)
    assert 98 < actions.mean() and actions.mean() < 102

def test_action_clipped():
    actions = narrow_skewed_strategy.generate_actions(np.random.randn(2), n=100)
    assert (99.9 <= actions).all() and (actions <= 100).all()

def test_log_probs_are_reasonable():
    assert GaussianSamplingStrategy.compute_log_probs(Normal(((0.,),), ((1,),)), (0.,)) > GaussianSamplingStrategy.compute_log_probs(Normal(((0.,),), ((1.,),)), ((1.,),))
    assert GaussianSamplingStrategy.compute_log_probs(Normal(((0.,),), ((1.,),)), ((0.,),)) > GaussianSamplingStrategy.compute_log_probs(Normal(((0.,),), ((2.,),)), ((0.,),))
    assert GaussianSamplingStrategy.compute_log_probs(Normal(((0.,),), ((1.,),)), ((0.,),)) > GaussianSamplingStrategy.compute_log_probs(Normal(((0.,),), ((1.,),)), ((1.,),))
    assert np.allclose(GaussianSamplingStrategy.compute_log_probs(Normal(((0.,),), ((1.,),)), ((0.,),)) * 2, GaussianSamplingStrategy.compute_log_probs(Normal(((0.,0.),), ((1.,1.),)), ((0.,0.),)))
    assert np.allclose(GaussianSamplingStrategy.compute_log_probs(Normal(((0.,),), ((1.,),)), ((0.5,),)), GaussianSamplingStrategy.compute_log_probs(Normal(((1.,),), ((1.,),)), ((0.5,),)))

def test_loss_function_is_reasonable():
    standard_normal = lambda batch: np.array(((0., 0.),))
    skewed = lambda batch: np.array(((1., 0.),))
    strategy = deepcopy(single_action_strategy)
    strategy.model = standard_normal
    assert np.allclose(strategy.compute_loss(None, ((0.,)), (1.)) * 2, strategy.compute_loss(None, ((0.,)), (2.,)))
    assert strategy.compute_loss(None, ((0.,)), (1.)) < strategy.compute_loss(None, ((1.,)), (1.,))
    strategy.model = skewed
    assert strategy.compute_loss(None, ((0.,)), (1.)) > strategy.compute_loss(None, ((1.,)), (1.,))
    assert np.allclose(strategy.compute_loss(None, ((.5,)), (1.)), strategy.compute_loss(None, ((.5,)), (1.,)))
    assert np.allclose(strategy.compute_loss(None, ((.5,)), (2.)), strategy.compute_loss(None, ((.5,)), (-2.,)) * -1)

def test_uses_training_config():
    strategy = GaussianSamplingStrategy(
        action_spec=wide_action_spec,
        observation_spec=wide_observation_spec
    )
    counter = EpochCounter()
    history = strategy.learn(
        training_history=[(
            np.random.choice([1, -1]),
            np.random.randn(2),
            np.random.randn(2),
            np.random.randn(),
        )] * 10,
        training_config=TrainingConfig(
            epochs=50,
            batch_size=100,
            patience=50,
            lr=1e-3,
            additional_callbacks=[counter]
        ),
        augmentation_function=Game.no_symmetries
    )
    assert isinstance(history, callbacks.History)
    assert counter.counter == 50
    assert strategy.model.optimizer._learning_rate == 1e-3
    assert history.epoch == list(range(50))

@probabilistic
def test_learns_linear_case():
    strategy = GaussianSamplingStrategy(
        observation_spec=wide_observation_spec,
        action_spec=BoundedArray(minimum=(0.,0.), maximum=(2., 2.), shape=(2,), dtype=np.float32)
    )
    strategy.learn(
        training_history=[
            (
                1.,
                np.array((i, i), dtype=float),
                np.array((j, j), dtype=float),
                np.array(1. if i == j else -.1, dtype=float)
            ) for i in range(3) for j in range(3)
        ] * 100,
        augmentation_function=Game.no_symmetries,
        training_config=TrainingConfig(
            epochs=10,
            validation_split=1,
            lr=1e-3
        )
    )

    for i in range(3):
        actions = strategy.generate_actions(observation=np.array((i, i), dtype=float), n=100)
        assert np.abs(actions.mean() - i) < 1
        assert actions.std() < 1.

@probabilistic
def test_learns_split_case():
    strategy = GaussianSamplingStrategy(
        observation_spec=wide_observation_spec,
        action_spec=BoundedArray(minimum=(0.,0.), maximum=(3., 3.), shape=(2,), dtype=np.float32)
    )
    strategy.learn(
        training_history=[
            (
                np.array(1. if i in (1, 2) else -.1, dtype=float),
                np.array((0, 0), dtype=float),
                np.array((i, i), dtype=float),
                1.,
            ) for i in range(4)
        ] * 100,
        augmentation_function=Game.no_symmetries,
        training_config=TrainingConfig(
            epochs=10,
            validation_split=1,
            lr=1e-3
        )
    )

    actions = strategy.generate_actions(observation=np.array((0, 0), dtype=float), n=100)
    assert 1 < actions.mean() and actions.mean() < 2
    assert .1 < actions.std() and actions.std() < 2
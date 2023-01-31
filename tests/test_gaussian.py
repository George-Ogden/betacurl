from src.sampling import GaussianSamplingStrategy

from dm_env.specs import BoundedArray
import tensorflow as tf
import numpy as np

single_action_spec = BoundedArray(minimum=(-100,), maximum=(100,), shape=(1,), dtype=np.float32)
single_observation_spec = BoundedArray(minimum=(-100,), maximum=(100,), shape=(1,), dtype=np.float32)
wide_action_spec = BoundedArray(minimum=(-100,-100), maximum=(100, 100), shape=(2,), dtype=np.float32)
wide_observation_spec = BoundedArray(minimum=(-100,-100), maximum=(100, 100), shape=(2,), dtype=np.float32)
wide_range_strategy = GaussianSamplingStrategy(action_spec=wide_action_spec, observation_spec=wide_observation_spec, latent_size=4)
single_action_strategy = GaussianSamplingStrategy(action_spec=single_action_spec, observation_spec=single_observation_spec, latent_size=4)

def test_samples_are_normal():
    means = np.ones(1000)
    log_stds = np.log(np.ones(1000) * 2)
    actions = np.concatenate((means, log_stds), axis=-1).reshape(-1, 4)
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        samples = wide_range_strategy.postprocess_actions(actions)
        mean = float(tf.reduce_mean(samples).numpy())
        std = float(tf.math.reduce_std(samples).numpy())
    assert .8 < mean < 1.2
    assert 1.5 < std < 2.5

def test_log_probs_are_reasonable():
    assert GaussianSamplingStrategy.compute_log_probs(((0.,),), ((0.,),), ((0.,),)) > GaussianSamplingStrategy.compute_log_probs(((0.,),), ((0.,),), ((1.,),))
    assert GaussianSamplingStrategy.compute_log_probs(((0.,),), ((0.,),), ((0.,),)) > GaussianSamplingStrategy.compute_log_probs(((0.,),), ((1.,),), ((0.,),))
    assert GaussianSamplingStrategy.compute_log_probs(((0.,),), ((0.,),), ((0.,),)) > GaussianSamplingStrategy.compute_log_probs(((0.,),), ((0.,),), ((1.,),))
    assert np.allclose(GaussianSamplingStrategy.compute_log_probs(((1.,),), ((0.,),), ((0.5,),)), GaussianSamplingStrategy.compute_log_probs(((0.,),), ((0.,),), ((0.5,),)))
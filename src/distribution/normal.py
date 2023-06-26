from __future__ import annotations

from tensorflow_probability import distributions, math, util
import tensorflow as tf
import numpy as np

from dm_env.specs import BoundedArray
from typing import List, Optional, Tuple, Union

from .config import NormalDistributionConfig
from .base import DistributionFactory

class ClippedNormalDistribution(distributions.Normal):
    def __init__(
        self,
        loc: tf.Tensor,
        scale: tf.Tensor,
        *args,
        **kwargs
    ):
        """either specify `bounds` or `lower_bound` and `upper_bound`"""
        # parameters must all have the same shape so the lower/upper bounds is a workaround
        
        if "bounds" in kwargs:
            bounds = kwargs.pop("bounds")
        elif len(args) == 1:
            bounds = args[0]
            args = ()
        elif len(args) == 2:
            bounds = tf.stack(args, axis=-1)
            args = ()
        else:
            assert "lower_bound" in kwargs and "upper_bound" in kwargs, "either `bounds` or `lower_bound` and `upper_bound` must be given"
            lower_bound = kwargs.pop("lower_bound")
            upper_bound = kwargs.pop("upper_bound")
            bounds = tf.stack((lower_bound, upper_bound), axis=-1)
        
        assert bounds.shape[-1] == 2, "the bounds must come in pairs (min-max)"
        assert bounds.shape[:-1] == loc.shape[1-bounds.ndim:], "the bounds must have the same batch size as the loc"
        
        super().__init__(loc=loc, scale=scale, **kwargs)
        
        self.bounds = tf.cast(bounds, dtype=self.dtype)
        
        lower_bound, upper_bound = tf.split(self.bounds, 2, axis=-1)
        self.lower_bound = tf.squeeze(lower_bound, axis=-1)
        self.upper_bound = tf.squeeze(upper_bound, axis=-1)

        self._parameters["lower_bound"] = self.lower_bound
        self._parameters["upper_bound"] = self.upper_bound
    
    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        return super()._parameter_properties(
            dtype=dtype,
            num_classes=num_classes
        ) | dict(
            lower_bound=util.ParameterProperties(),
            upper_bound=util.ParameterProperties()
        )
    
    def _sample_n(self, n, seed=None):
        sample = super()._sample_n(n, seed=seed)
        return tf.clip_by_value(
            sample,
            tf.gather(self.bounds, 0, axis=-1),
            tf.gather(self.bounds, 1, axis=-1)
        )

class NormalDistributionFactory(DistributionFactory):
    CONFIG_CLASS = NormalDistributionConfig
    def __init__(
        self,
        move_spec: BoundedArray,
        config: NormalDistributionConfig = NormalDistributionConfig()
    ):
        super().__init__(move_spec, config=config)

    def _create_distribution(
        self,
        parameters: tf.Tensor,
        features: Optional[tf.Tensor] = None
    ) -> distributions.Normal:
        parameters += tf.random.normal(parameters.shape) * self.noise_ratio
        mean, std = tf.split(parameters, 2, axis=-1)
        
        std = tf.squeeze(std, axis=-1)
        std = tf.nn.softplus(std) + 1e-5

        mean = tf.squeeze(mean, axis=-1)
        mean = tf.nn.sigmoid(mean)
        mean *= self.action_range[1] - self.action_range[0]
        mean += self.action_range[0]

        bounds = self.generate_bounds(parameters)

        return ClippedNormalDistribution(
            loc=mean,
            scale=std,
            bounds=bounds
        )

    @property
    def parameters_shape(self) -> Tuple[int, ...]:
        return self.action_shape + (2,)

    def parameterize(
        self,
        actions: Union[tf.Tensor, np.ndarray]
    ) -> tf.Tensor:
        return tf.stack(
            [
                (   
                    actions - self.action_range[0]
                ) / (
                    self.action_range[1] - self.action_range[0]
                ),
                math.softplus_inverse(tf.ones_like(actions))
            ],
            axis=-1
        )

    def compute_loss(
        self,
        target_parameters: tf.Tensor,
        parameters: distributions.Distribution
    ) -> tf.Tensor:
        raise NotImplementedError("not implemented yet")
    
    @staticmethod
    def aggregate_parameters(
        parameters: List[Tuple[tf.Tensor, int]]
    ) -> tf.Tensor:
        parameters, counts = zip(*parameters)
        parameters = tf.cast(tf.stack(parameters, axis=-1), dtype=tf.float32)
        
        means, stds = tf.split(parameters, 2, axis=-2)
        means = tf.squeeze(means, axis=-2)
        stds = tf.squeeze(stds, axis=-2)
        stds = tf.nn.softplus(stds) + 1e-5
        variances = tf.square(stds)

        counts = tf.constant(counts, dtype=tf.float32)

        return tf.stack(
            (
                tf.reduce_sum(
                    means * counts,
                    axis=-1
                ) / tf.reduce_sum(counts),
                math.softplus_inverse(
                    tf.sqrt(
                        tf.reduce_sum(variances * counts, axis=-1) / tf.reduce_sum(counts)
                    )
                )
            ),
            axis=-1
        )
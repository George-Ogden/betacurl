import tensorflow as tf
import numpy as np
import wandb

def pytest_configure(config):
    # disabel wandb logging
    wandb.init("test run", mode="disabled")

    # run tests without GPU
    tf.config.experimental.set_visible_devices([], "GPU")

    # seed for repeatability
    np.random.seed(0)
    tf.random.set_seed(0)
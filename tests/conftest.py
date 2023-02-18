import tensorflow as tf
import wandb

def pytest_configure(config):
    # disabel wandb logging
    wandb.init("test run", mode="disabled")

    # run tests without GPU
    tf.config.experimental.set_visible_devices([], "GPU")
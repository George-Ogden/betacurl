from src.io.io import SaveableObject
import os

from tensorflow.keras.models import load_model
from tensorflow.keras import callbacks, optimizers
import tensorflow as tf
import numpy as np

from wandb.keras import WandbMetricsLogger
from tqdm.keras import TqdmCallback

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass
class TrainingConfig:
    epochs: int = 20
    """number of epochs to train each model for"""
    batch_size: int = 64
    """training batch size"""
    patience: int = 7
    """number of epochs without improvement during training (0 to ignore)"""
    lr: float = 1e-2
    """model learning rate"""
    validation_split: float = 0.1
    """proportion of data to validate on"""
    loss: str = "mse"
    optimizer_type: str = "Adam"
    metrics: List[str] = ["mae"]
    callbacks: Optional[List[callbacks.Callback]] = None
    compile_kwargs: Optional[Dict[str, Any]] = None
    fit_kwargs: Optional[Dict[str, Any]] = None

    @property
    def optimizer(self) -> optimizers.Optimizer:
        optimizers.get(self.optimizer_type)(
            learning_rate=self.lr, 
            **(self.optimizer_kwargs or {})
        )

class ModelDecorator(SaveableObject):
    model: tf.keras.Model = None
    DEFAULT_MODEL_FILE: str = "model.h5"
    def save(self, directory: str):
        model = self.model
        self.model = None

        super().save(directory)
        model.save(self.get_model_filename(directory))

        self.model = model

    @classmethod
    def load(cls, directory: str) -> "Self":
        model = super().load(directory)
        model.model = load_model(cls.get_model_filename(directory))
        return model

    @classmethod
    def get_model_filename(cls, directory):
        return os.path.join(directory,cls.DEFAULT_MODEL_FILE)

    def learn(self, training_data):
        ...

    def normalise_inputs(self, inputs: np.ndarray) -> np.ndarray:
        return inputs

    def normalise_outputs(self, outputs: np.ndarray) -> np.ndarray:
        return outputs

    def fit(self, X: np.ndarray, Y: np.ndarray, training_config: TrainingConfig) -> callbacks.History:
        assert type(X) == np.ndarray and type(Y) == np.ndarray
        compile_options = {
            "optimizer": training_config.optimizer,
            "loss": training_config.loss,
            "metrics": training_config.metrics,
        }

        train_options = {
            "batch_size": training_config.batch_size,
            "validation_split": training_config.validation_split,
            "verbose": 0,
            "callbacks": [
                WandbMetricsLogger(),
                TqdmCallback(desc=f"Training {type(self).__name__} ({self.model._name})"),
            ] + (training_config.callbacks or []) + ([callbacks.EarlyStopping(patience=training_config.patience, monitor="val_mae")] if training_config.patience > 0 else []),
            "epochs": training_config.epochs,
        }

        X = self.normalise_inputs(X)
        Y = self.normalise_outputs(Y)
        self.model.compile(**compile_options)
        return self.model.fit(X, Y, **{**train_options, **(training_config.fit_kwargs or {})})
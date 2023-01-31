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
    optimizer: str = "Adam"
    learning_rate: float = 1e-2
    loss: str = "mse"
    metrics: List[str] = ["mae"]
    batch_size: int = 64
    patience: int = 7
    validation_split: float = 0.1
    callbacks: Optional[List[callbacks.Callback]] = None
    epochs: int = 20
    compile_kwargs: Optional[Dict[str, Any]] = None
    fit_kwargs: Optional[Dict[str, Any]] = None

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
            "optimizer": optimizers.get(training_config.optimizer)(
                learning_rate=training_config.learning_rate, 
                **(training_config.optimizer_kwargs or {})
            ),
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
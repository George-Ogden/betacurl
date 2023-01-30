from src.io.io import SaveableObject
import os

from tensorflow.keras.models import load_model
from tensorflow.keras import callbacks, optimizers
import tensorflow as tf
import numpy as np

from typing import Any
from wandb.keras import WandbMetricsLogger
# from typing import Self

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

    def fit(self, X: np.ndarray, Y: np.ndarray, **kwargs: Any) -> callbacks.History:
        assert type(X) == np.ndarray and type(Y) == np.ndarray
        compile_options = {
            "optimizer": optimizers.Adam(learning_rate=kwargs.pop("lr", 1e-2)),
            "loss": "mse",
            "metrics": ["mae"]
        }
        train_options = {
            "batch_size": 64,
            "validation_split": 0.1,
            "verbose": 1,
            "callbacks": [callbacks.EarlyStopping(patience=kwargs.pop("patience", 5), monitor="val_mae"), WandbMetricsLogger()],
            "epochs": 50,
        }

        used_kwargs = []
        for k, v in kwargs.items():
            if k in compile_options:
                compile_options[k] = v
                used_kwargs.append(k)

        for key in used_kwargs:
            del kwargs[key]

        X = self.normalise_inputs(X)
        Y = self.normalise_outputs(Y)
        self.model.compile(**compile_options)
        return self.model.fit(X, Y, **train_options | kwargs)
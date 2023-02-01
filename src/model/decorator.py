from tensorflow.keras import callbacks
import numpy as np

from abc import abstractmethod, ABCMeta

from ..io import SaveableModel

from .config import TrainingConfig

class Learnable(metaclass=ABCMeta):
    @abstractmethod
    def learn(self, training_data):
        ...

class ModelDecorator(SaveableModel, Learnable):
    def normalise_inputs(self, inputs: np.ndarray) -> np.ndarray:
        return inputs

    def normalise_outputs(self, outputs: np.ndarray) -> np.ndarray:
        return outputs

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        training_config: TrainingConfig = TrainingConfig()
    ) -> callbacks.History:
        if type(X) != np.ndarray:
            X = np.array(X)
        if type(Y) != np.ndarray:
            Y = np.array(Y)

        compile_options = {
            "optimizer": training_config.optimizer,
            "loss": training_config.loss,
            "metrics": training_config.metrics,
            **(training_config.compile_kwargs or {})
        }

        train_options = {
            "batch_size": training_config.batch_size,
            "validation_split": training_config.validation_split,
            "verbose": 0,
            "callbacks": training_config.callbacks,
            "epochs": training_config.epochs,
            **(training_config.fit_kwargs or {})
        }

        X = self.normalise_inputs(X)
        Y = self.normalise_outputs(Y)
        self.model.compile(**compile_options)
        return self.model.fit(X, Y, **train_options)
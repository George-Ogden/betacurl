from tensorflow.keras import callbacks, utils
from copy import copy, deepcopy
from tensorflow import data
import tensorflow as tf
import numpy as np

from typing import Any, Dict, List, Tuple, Union
from abc import abstractmethod, ABCMeta

from ..utils import SaveableModel

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

    def compile_model(self, training_config: TrainingConfig):
        compile_options = {
            "optimizer": training_config.optimizer,
            "loss": training_config.loss,
            "metrics": training_config.metrics,
            **(training_config.compile_kwargs or {})
        }
        self.model.compile(**compile_options)

    @staticmethod
    def merge(*items: List[Any]):
        assert len(items) > 0
        shared_item = deepcopy(items[0])
        using_dict = isinstance(shared_item, dict)
        for item in items[1:]:
            for k, w in (item if using_dict else vars(item)).items():
                v = shared_item.get(k, None) if using_dict else getattr(shared_item, k, None)
                if v is None:
                    v = w
                else:
                    # "Irrefutable pattern is allowed only for the last case statement"
                    # when matching `type(v)`
                    match type(v).__name__:
                        case "bool":
                            v |= w
                        case "list":
                            v.extend(w)
                        case "int":
                            v += w
                        case "dict":
                            v = ModelDecorator.merge(v, w)
                if using_dict:
                    shared_item[k] = v
                else:
                    setattr(shared_item, k, v)
        return shared_item

    @staticmethod
    def create_dataset(dataset: List[Tuple[Union[np.ndarray, float]]]) -> data.Dataset:
        transposed_data = tuple(np.array(data, dtype=np.float32) for data in zip(*dataset))
        return data.Dataset.from_tensor_slices(transposed_data)

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        training_config: TrainingConfig = TrainingConfig()
    ) -> callbacks.History:
        X, Y, train_options = self.pre_fit(X, Y, training_config)
        return self.model.fit(X, Y, **train_options)

    def pre_fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        training_config: TrainingConfig,
        convert_to_numpy:bool=True
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """prepare inputs for Model.fit style args

        Returns:
            Tuple[np.ndarray, np.ndarray, Dict[str, Any]]: X, Y, kwargs
        """
        if convert_to_numpy:
            if type(X) != np.ndarray:
                X = np.array(X, dtype=np.float32)
            if type(Y) != np.ndarray:
                Y = np.array(Y, dtype=np.float32)

        self.compile_model(training_config)

        train_options = {
            "batch_size": training_config.batch_size,
            "validation_split": training_config.validation_split,
            "verbose": training_config.verbose,
            "callbacks": training_config.callbacks,
            "epochs": training_config.training_epochs,
            **(training_config.fit_kwargs or {})
        }

        X = self.normalise_inputs(X)
        Y = self.normalise_outputs(Y)
        return X, Y, train_options

class CustomDecorator(ModelDecorator):
    @abstractmethod
    def compute_loss(self, *batch: List[tf.Tensor]) -> tf.Tensor:
        ...

    def train_step(self, batch: np.ndarray, optimizer: tf.optimizers.Optimizer) -> np.ndarray:
        with tf.GradientTape() as tape:
            loss = self.compute_loss(*batch)
        grads = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients([
            (
                tf.where(tf.math.is_nan(grad),tf.zeros_like(grad), grad),
                var
            ) for grad, var in
                zip(
                    grads,
                    self.model.trainable_variables
                ) if grad is not None
        ])
        return loss

    def fit(self, dataset: data.Dataset, training_config: TrainingConfig = TrainingConfig()) -> callbacks.History:
        training_config = copy(training_config)
        train_dataset, val_dataset = utils.split_dataset(dataset, right_size=training_config.validation_split, shuffle=True)
        batch_size = training_config.batch_size
        training_config.metrics = []

        self.compile_model(training_config)
        optimizer = self.model.optimizer

        history = callbacks.History()
        callback = callbacks.CallbackList(
            training_config.callbacks + [history],
            model=self.model,
            add_history=False,
            add_progbar=training_config.verbose != 0,
            verbose=training_config.verbose,
            epochs=training_config.training_epochs,
            steps=(len(train_dataset) - 1) // training_config.batch_size + 1,
        )

        self.model.stop_training = False
        callback.on_train_begin()
        for epoch in range(training_config.training_epochs):
            callback.on_epoch_begin(epoch)
            loss = 0
            for step, batch in enumerate(train_dataset.batch(batch_size)):
                callback.on_train_batch_begin(step)
                loss += self.train_step(batch, optimizer)
                callback.on_train_batch_end(step)
            val_loss = 0
            for step, batch in enumerate(val_dataset.batch(batch_size)):
                val_loss += self.compute_loss(*batch)
            callback.on_epoch_end(epoch, {"loss": loss / len(train_dataset), "val_loss": val_loss / len(val_dataset)})
            if self.model.stop_training:
                break
        callback.on_train_end()
        return history
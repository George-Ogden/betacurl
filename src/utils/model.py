from tensorflow.keras.models import load_model, save_model
from tensorflow import keras
import os

from typing import Dict

from .io import SaveableObject

class SaveableModel(SaveableObject):
    model: keras.Model = None
    DEFAULT_FILENAME = "model.pickle"
    DEFAULT_MODEL_FILE: str = "model.h5"
    def save(self, directory: str):
        model = self.model
        self.model = None

        super().save(directory)
        if model is not None:
            save_model(model, self.get_model_filename(directory))

        self.model = model

    @classmethod
    def load(cls, directory: str) -> "Self":
        model = super().load(directory)
        if os.path.exists(model.get_model_filename(directory)):
            model.model = load_model(model.get_model_filename(directory))
        else:
            model.model = None
        return model

    @classmethod
    def get_model_filename(cls, directory):
        return os.path.join(directory,cls.DEFAULT_MODEL_FILE)

class SaveableMultiModel(SaveableModel):
    DEFAULT_FILENAME = "model.pickle"
    MODELS: Dict[str, str] = {} # dictionary from variable name to filename
    def save(self, directory: str):
        models = {}
        for attr in self.MODELS:
            models[attr] = getattr(self, attr)
            setattr(self, attr, None)

        super().save(directory)

        for attr, filename in self.MODELS.items():
            if models[attr] is not None:
                save_model(models[attr], os.path.join(directory, filename))

            setattr(self, attr, models[attr])

    @classmethod
    def load(cls, directory: str) -> "Self":
        model = super().load(directory)
        for attr, filename in model.MODELS.items():
            if os.path.exists(os.path.join(directory, filename)):
                setattr(
                    model,
                    attr,
                    load_model(os.path.join(directory, filename))
                )
            else:
                setattr(model, attr, None)
        return model
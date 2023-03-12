from __future__ import annotations

import pickle
import os

from typing import List

class SaveableObject:
    DEFAULT_FILENAME: str = None
    SEPARATE_ATTRIBUTES: List[str] = [] # names of attributes to be saved separately
    def save(self, directory: str):
        if not os.path.exists(directory):
            os.mkdir(directory)

        separate_objects: List[SaveableObject] = [getattr(self, sub_object) for sub_object in self.SEPARATE_ATTRIBUTES]

        for attribute, object in zip(self.SEPARATE_ATTRIBUTES, separate_objects):
            setattr(self, attribute, type(object))

        with open(self.get_path(directory), "wb") as f:
            pickle.dump(self, f)

        for object in separate_objects:
            if object is not None:
                object.save(directory)

        for attribute, object in zip(self.SEPARATE_ATTRIBUTES, separate_objects):
            setattr(self, attribute, object)

    @classmethod
    def load(cls, directory: str) -> "Self":
        path = cls.get_path(directory)
        with open(path, "rb") as f:
            object = pickle.load(f)
        
        for attribute in cls.SEPARATE_ATTRIBUTES:
            if getattr(object, attribute) is type(None):
                setattr(
                    object,
                    attribute,
                    None
                )
            else:
                setattr(
                    object,
                    attribute,
                    getattr(object, attribute).load(directory)
                )
        return object

    @classmethod
    def get_path(cls, directory):
        return os.path.join(directory, cls.DEFAULT_FILENAME)
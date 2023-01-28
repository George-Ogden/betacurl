import pickle
import os

# from typing import Self

class SaveableObject:
    DEFAULT_FILENAME: str = None
    def save(self, directory: str):
        if not os.path.exists(directory):
            os.mkdir(directory)
        with open(self.get_path(directory), "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, directory: str) -> "Self":
        path = cls.get_path(directory)
        with open(path, "rb") as f:
            return pickle.load(f)

    @classmethod
    def get_path(cls, directory):
        return os.path.join(directory, cls.DEFAULT_FILENAME)
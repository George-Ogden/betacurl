from dataclasses import asdict

from typing import List
import argparse

class Config:
    def keys(self) -> List[str]:
        return self.__match_args__

    def __getitem__(self, key):
        return getattr(self, key)

    def set_args(self, args: argparse.Namespace) -> "Self":
        for attribute in asdict(self):
            value = getattr(self, attribute)
            if isinstance(value, Config):
                setattr(self, attribute, type(value).from_args(args))
            elif attribute in args:
                setattr(self, attribute, getattr(args, attribute))
        return self
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "Self":
        config = cls()
        return config.set_args(args)
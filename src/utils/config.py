from typing import List

class Config:
    def keys(self) -> List[str]:
        return self.__match_args__

    def __getitem__(self, key):
        return getattr(self, key)
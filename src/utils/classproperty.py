from typing import Any, Callable, Type

class classproperty:
    def __init__(self, f: Callable[[Type], Any]) -> None:
        self.f = classmethod(f)

    def __get__(self, instance: Any, owner: Type) -> Any:
        return self.f.__get__(None, owner)()
from dataclasses import dataclass
from typing import ClassVar

from src.utils import Config

def test_config_with_all_attrs():
    @dataclass
    class CustomConfig(Config):
        name: str = "hello"
        number: int = 12
    
    assert dict(**CustomConfig()) == dict(name="hello", number=12)

def test_config_with_restricted_attrs():
    @dataclass
    class CustomConfig(Config):
        name: str = "hello"
        number: int = 12
        cls_attr: ClassVar[float] = .5
    
    assert dict(**CustomConfig()) == dict(name="hello", number=12)
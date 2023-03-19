from dataclasses import dataclass
from typing import ClassVar

from src.utils.parser import get_dataclass_attributes_doc
from src.utils.config import Config
from src.utils import create_parser

@dataclass
class SimpleConfig(Config):
    a: str = "a"
    """A"""
    b: int = 2
    c: ClassVar[float] = 3.

@dataclass
class ComplexConfig(Config):
    x: str = "x"
    """X"""
    y: int = 2
    """Y"""
    z: float = 3.
    """Z"""
    simple_config: SimpleConfig = SimpleConfig()

def test_doc():
    doc = get_dataclass_attributes_doc(SimpleConfig)
    assert doc["a"] == "A"
    assert not "b" in doc
    assert not "c" in doc

    doc = get_dataclass_attributes_doc(ComplexConfig)
    assert doc["x"] == "X"
    assert doc["y"] == "Y"
    assert doc["z"] == "Z"

def test_create_parser():
    parser = create_parser(ComplexConfig())
    args = parser.parse_args([])
    assert args.a == "a"
    assert args.x == "x"
    assert args.y == 2
    assert args.z == 3.
    for arg in ("b","c"):
        assert not arg in args

def test_create_parser_with_additional_args():
    parser = create_parser(
        SimpleConfig(),
        additional_arguments=[
            ("project_name", "test")
        ]
    )
    args = parser.parse_args([])
    assert args.a == "a"
    assert args.project_name == "test"
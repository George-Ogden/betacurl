from dataclasses import dataclass
from typing import ClassVar

from src.utils import ParserBuilder
from src.utils.config import Config

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
    doc = ParserBuilder.get_dataclass_attributes_doc(SimpleConfig)
    assert doc["a"] == "A"
    assert not "b" in doc
    assert not "c" in doc

    doc = ParserBuilder.get_dataclass_attributes_doc(ComplexConfig)
    assert doc["x"] == "X"
    assert doc["y"] == "Y"
    assert doc["z"] == "Z"

def test_create_parser():
    parser = ParserBuilder().add_dataclass(ComplexConfig()).build()
    args = parser.parse_args([])
    assert args.a == "a"
    assert args.x == "x"
    assert args.y == 2
    assert args.z == 3.
    for arg in ("b","c"):
        assert not arg in args

def test_create_parser_with_additional_args(capsys):
    parser = ParserBuilder().add_dataclass(
        SimpleConfig()
    ).add_argument(
        key="project_name",
        value="DEFAULT VALUE",
        help="TEST PROJECT HELP",
    ).build()
    parser.print_help()

    captured = capsys.readouterr()
    assert "default: DEFAULT VALUE" in captured.out
    assert "TEST PROJECT HELP" in captured.out

    args = parser.parse_args([])
    assert args.a == "a"
    assert args.project_name == "DEFAULT VALUE"
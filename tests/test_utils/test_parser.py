from dataclasses import dataclass
from typing import ClassVar

from src.utils import ParserBuilder
from src.utils.config import Config

@dataclass
class SimpleConfig(Config):
    a: str = "a"
    """A"""
    b: int = 2
    """B"""
    c: float = 3.

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
    assert doc["b"] == "B"
    assert not "c" in doc

    doc = ParserBuilder.get_dataclass_attributes_doc(ComplexConfig)
    assert doc["x"] == "X"
    assert doc["y"] == "Y"
    assert doc["z"] == "Z"

def test_create_parser():
    parser = ParserBuilder().add_dataclass(ComplexConfig()).build()
    args = parser.parse_args([])

    assert args.a == "a"
    assert args.b == 2
    assert args.x == "x"
    assert args.y == 2
    assert args.z == 3.
    assert not "c" in args

def test_create_parser_with_additional_args(capsys):
    parser = ParserBuilder().add_dataclass(
        SimpleConfig()
    ).add_argument(
        name="project_name",
        default="DEFAULT VALUE",
        help="TEST PROJECT HELP",
    ).build()
    parser.print_help()

    captured = capsys.readouterr()
    assert "default: DEFAULT VALUE" in captured.out
    assert "TEST PROJECT HELP" in captured.out

    args = parser.parse_args([])
    assert args.a == "a"
    assert args.project_name == "DEFAULT VALUE"

def test_config_set_args():
    simple_config = SimpleConfig(a="A")
    args = ParserBuilder().add_dataclass(simple_config).build().parse_args(["--b", "8"])
    simple_config.set_args(args)

    assert simple_config.a == "A"
    assert simple_config.b == 8
    assert simple_config.c == 3.

    complex_config = ComplexConfig(x="X", simple_config=simple_config)
    args = ParserBuilder().add_dataclass(complex_config).build().parse_args(["--b", "6", "--z", ".9"])
    complex_config.set_args(args)

    assert complex_config.x == "X"
    assert complex_config.y == 2
    assert complex_config.z == .9

    simple_config = complex_config.simple_config
    assert simple_config.a == "A"
    assert simple_config.b == 6
    assert simple_config.c == 3.

def test_config_from_args():
    simple_config = SimpleConfig(a="A", c=4.)
    args = ParserBuilder().add_dataclass(simple_config).build().parse_args(["--b", "8"])
    simple_config = SimpleConfig.from_args(args)

    assert simple_config.a == "A"
    assert simple_config.b == 8
    assert simple_config.c == 3.

    complex_config = ComplexConfig(x="X", simple_config=simple_config)
    args = ParserBuilder().add_dataclass(complex_config).build().parse_args(["--b", "6", "--z", ".9"])
    complex_config = ComplexConfig.from_args(args)

    assert complex_config.x == "X"
    assert complex_config.y == 2
    assert complex_config.z == .9

    simple_config = complex_config.simple_config
    assert simple_config.a == "A"
    assert simple_config.b == 6
    assert simple_config.c == 3.
from simple_parsing.docstring import get_attribute_docstring
from dataclasses import asdict, is_dataclass
from typing import get_type_hints
import argparse

from typing import Any, Optional, List, Tuple, Type

from .config import Config

# modified from https://stackoverflow.com/a/66239222/12103577
def get_dataclass_attributes_doc(config: Type[Config]):
    def get_attribute_unified_doc(some_dataclass: Type[Config], key: str) -> str:
        """returns a string that chains the above-comment, inline-comment and docstring"""
        all_docstrings = get_attribute_docstring(some_dataclass, key)
        doc_list = asdict(all_docstrings).values()
        return '\n'.join(doc_list).strip()

    attribute_docs = {}
    for key in get_type_hints(config).keys():
        doc = get_attribute_unified_doc(config, key)
        if len(doc):
            attribute_docs[key] = doc
    return attribute_docs

def create_parser(config: Config, additional_arguments: Optional[List[Tuple[str, Any]]] = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    attribute_docs = {
        "project_name": "wandb project name"
    }
    def add_argument(key: str, value: Any):
        additional_options = {}
        if type(value) == bool:
            additional_options["action"] = "store_true"
        else:
            additional_options["type"] = type(value)
        parser.add_argument(f"--{key}", default=value, help=attribute_docs[key], **additional_options)

    def add_dataclass(config):
        nonlocal attribute_docs
        attribute_docs |= get_dataclass_attributes_doc(type(config))
        for attribute in asdict(config):
            value = getattr(config, attribute)
            if is_dataclass(value):
                add_dataclass(value)
            else:
                if len(attribute_docs.get(attribute, "")) > 0:
                    add_argument(attribute, value)
    add_dataclass(config)
    if additional_arguments:
        for key, value in additional_arguments:
            add_argument(key, value)
    return parser
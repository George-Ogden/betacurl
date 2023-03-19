from simple_parsing.docstring import get_attribute_docstring
from dataclasses import asdict, is_dataclass
from typing import get_type_hints
import argparse

from typing import Any, Optional, Type

from .config import Config

class ParserBuilder:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.attribute_docs = {}

    def add_dataclass(self, config: Config) -> "Self":
        self.attribute_docs |= self.get_dataclass_attributes_doc(type(config))
        for attribute in asdict(config):
            value = getattr(config, attribute)
            if is_dataclass(value):
                self.add_dataclass(value)
            else:
                help = self.attribute_docs.get(attribute, "")
                if len(help) > 0:
                    self.add_argument(attribute, value, help)
        return self

    def add_argument(self, key: str, value: Any, help: Optional[str] = None, **additional_options) -> "Self":
        if type(value) == bool:
            additional_options["action"] = "store_true"
        else:
            additional_options["type"] = type(value)
        self.parser.add_argument(f"--{key}", default=value, help=help, **additional_options)
        return self

    @staticmethod
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

    def build(self) -> argparse.ArgumentParser:
        return self.parser
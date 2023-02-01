from src.game import Coach, CoachConfig, SamplingEvaluatingPlayerConfig
from src.sampling import RandomSamplingStrategy
from src.curling import CURLING_GAME
import wandb

from simple_parsing.docstring import get_attribute_docstring
from typing import get_type_hints
from dataclasses import asdict, is_dataclass
import argparse

def main(args):
    wandb.init(project=args.project_name, dir=args.save_directory)
    wandb.config.update(args)

    player_config = SamplingEvaluatingPlayerConfig(
        **{k: v for k, v in filter(lambda attr: hasattr(SamplingEvaluatingPlayerConfig, attr[0]), vars(args).items())}
    )
    coach_config = CoachConfig(
        player_config=player_config,
        **{k: v for k, v in filter(lambda attr: hasattr(CoachConfig, attr[0]), vars(args).items())}
    )

    coach = Coach(
        game=CURLING_GAME,
        config=coach_config,
        SamplingStrategyClass=RandomSamplingStrategy,
    )
    coach.learn()

# https://stackoverflow.com/a/66239222/12103577
def get_dataclass_attributes_doc(some_dataclass):
    def get_attribute_unified_doc(some_dataclass, key):
        """returns a string that chains the above-comment, inline-comment and docstring"""
        all_docstrings = get_attribute_docstring(some_dataclass, key)
        doc_list = asdict(all_docstrings).values()
        return '\n'.join(doc_list).strip()

    attribute_docs = {}
    for key in get_type_hints(some_dataclass).keys():
        attribute_docs[key] = get_attribute_unified_doc(some_dataclass, key)
    return attribute_docs

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    attribute_docs = {
        "project_name": "wandb project name"
    }
    def add_argument(key, value):
        additional_options = {}
        if type(value) == bool:
            additional_options["action"] = "store_true"
        else:
            additional_options["type"] = type(value)
        parser.add_argument(f"--{key}", default=value, help=attribute_docs[key], **additional_options)

    def add_dataclass(config):
        nonlocal attribute_docs
        attribute_docs |= get_dataclass_attributes_doc(type(config))
        for k, v in asdict(config).items():
            if is_dataclass(v):
                add_dataclass(v)
            else:
                if len(attribute_docs.get(k, "")) > 0:
                    add_argument(k, v)
    add_dataclass(CoachConfig())
    add_argument("project_name", "test project")
    return parser

if __name__ == "__main__":
    args = create_parser().parse_args()
    main(args)
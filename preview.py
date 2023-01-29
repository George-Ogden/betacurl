from src.game import SamplingEvaluatingPlayer, CoachConfig, Arena
from src.curling import CURLING_GAME

import argparse
import os

def main(args):
    player = SamplingEvaluatingPlayer.load(args.model_directory)
    player.num_eval_samples = 50
    arena = Arena(players=[player.dummy_constructor] * 2, game=CURLING_GAME)
    arena.play_game(display=True, training=False)

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_directory", default=os.path.join(CoachConfig.save_directory, CoachConfig.best_checkpoint_path))
    return parser

if __name__ == "__main__":
    args = create_parser().parse_args()
    main(args)
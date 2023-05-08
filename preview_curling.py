from src.player import NNMCTSPlayer
from src.game import CURLING_GAME
from src.coach import CoachConfig
from src.player import Arena

import argparse
import os

def main(args):
    player = NNMCTSPlayer.load(args.model_directory)
    player.widen()
    player.simulations = 50
    arena = Arena(players=[player.dummy_constructor, NNMCTSPlayer], game=CURLING_GAME)
    arena.players[1].eval_simulations = 50
    for i in range(10):
        print(arena.play_game(display=True, training=False, starting_player=i % 2))

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_directory", default=os.path.join(CoachConfig.save_directory, CoachConfig.last_checkpoint_path))
    return parser

if __name__ == "__main__":
    args = create_parser().parse_args()
    main(args)

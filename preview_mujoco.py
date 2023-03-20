from dm_control import viewer

from src.player import Arena, NNMCTSPlayer
from src.coach import CoachConfig
from src.game import MujocoGame

import argparse
import os

def main(args):
    game = MujocoGame(domain_name=args.domain_name, task_name=args.task_name)
    player = NNMCTSPlayer.load(args.model_directory)
    player.simulations = 50
    player.eval()
    viewer.launch(game.env, policy=lambda observation: player.move(game))

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_directory", default=os.path.join(CoachConfig.save_directory, CoachConfig.best_checkpoint_path))
    parser.add_argument("--domain_name", default="cartpole")
    parser.add_argument("--task_name", default="swingup")
    return parser

if __name__ == "__main__":
    args = create_parser().parse_args()
    main(args)
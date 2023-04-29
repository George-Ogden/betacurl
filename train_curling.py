import wandb

from src.coach import Coach, CoachConfig
from src.mcts import PolicyMCTSModel
from src.utils import ParserBuilder
from src.game import CURLING_GAME

def main(args):
    wandb.init(project=args.project_name, dir=args.save_directory)
    wandb.config.update(args)

    coach_config = CoachConfig.from_args(args)

    coach = Coach(
        game=CURLING_GAME,
        ModelClass=PolicyMCTSModel,
        config=coach_config
    )
    coach.learn()

if __name__ == "__main__":
    parser = ParserBuilder().add_dataclass(
        CoachConfig()
    ).add_argument(
        name="project_name",
        default="test project",
        help="wandb project name"
    ).build()
    args = parser.parse_args()
    main(args)

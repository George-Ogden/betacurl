import wandb

from src.coach import SinglePlayerCoach, SinglePlayerCoachConfig
from src.mcts import PolicyMCTSModel
from src.utils import ParserBuilder
from src.game import MujocoGame

def main(args):
    wandb.init(project=args.project_name, dir=args.save_directory)
    wandb.config.update(args)

    env = MujocoGame(
        domain_name=args.domain_name,
        task_name=args.task_name,
    )

    coach_config = SinglePlayerCoachConfig.from_args(args)
    coach = SinglePlayerCoach(
        game=env,
        config=coach_config,
        ModelClass=PolicyMCTSModel,
    )
    coach.learn()

if __name__ == "__main__":
    parser = ParserBuilder().add_dataclass(
        SinglePlayerCoachConfig()
    ).add_argument(
        name="domain_name",
        help="mujoco domain",
        required=True
    ).add_argument(
        name="task_name",
        help="mujoco task",
        required=True
    ).add_argument(
        name="project_name",
        default="test project",
        help="wandb project name"
    ).build()
    args = parser.parse_args()
    main(args)

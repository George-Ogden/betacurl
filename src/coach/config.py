from dataclasses import dataclass
from typing import ClassVar

from ..player import NNMCTSPlayerConfig
from ..model import TrainingConfig
from ..utils import Config

@dataclass
class CoachConfig(Config):
    resume_from_checkpoint: bool = False
    """continue training from previous checkpoint"""
    num_games_per_episode: int = 25
    """number of self-play games per model update"""
    num_iterations: int = 100
    """total number of training iterations"""
    evaluation_games: int = 10
    """number of games to determine best model"""
    win_threshold: float = .65
    """proportion of games that a new model must win to be considered the best"""
    save_directory: str = "output"
    """directory to save logs, model, files, etc. to"""
    best_checkpoint_path: str = "model-best"
    """name of best model"""
    successive_win_requirement: int = 7
    """number of games won by best model before training terminates"""
    num_eval_simulations: int = 50
    """number of simulations to run when evaluating"""
    model_filenames: str = "model-{:06}"
    player_config: NNMCTSPlayerConfig = NNMCTSPlayerConfig()
    training_config: TrainingConfig = TrainingConfig()

@dataclass
class PPOCoachConfig(CoachConfig):
    evaluation_games: int = 5
    """number of episodes to evalute on"""
    win_threshold: ClassVar[float] = 0.
from dataclasses import dataclass
from typing import ClassVar

from ..player import NNMCTSPlayerConfig
from ..model import TrainingConfig
from ..utils import Config

@dataclass
class CoachConfig(Config):
    warm_start_games: int = 2000
    """number of iterations to warm start with"""
    resume_from_checkpoint: bool = False
    """continue training from previous checkpoint"""
    num_games_per_episode: int = 5
    """number of self-play games per model update"""
    num_iterations: int = 100
    """total number of training iterations"""
    save_directory: str = "output"
    """directory to save logs, model, files, etc. to"""
    last_checkpoint_path: str = "model-last"
    """path to save last model"""
    model_filenames: str = "model-{:06}"
    save_frequency: int = 1
    """number of iterations between saving model checkpoints"""
    # values from muzero and alphazero papers
    initial_lr: float = 2e-2
    final_lr: float = 2e-4
    initial_temperature: float = 1.0
    final_temperature: float = 0.25
    player_config: NNMCTSPlayerConfig = NNMCTSPlayerConfig()
    training_config: TrainingConfig = TrainingConfig()

@dataclass
class SinglePlayerCoachConfig(CoachConfig):
    warm_start_games: ClassVar[int] = 0
    eval_games: int = 5
    """number of games to run for evaluation"""
    gae_lambda: float = 0.95
    eval_simulations: int = 15
    """number of simulations to run for evaluation"""
    best_checkpoint_path: str = "model-best"
    """path to save best model"""
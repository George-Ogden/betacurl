from ..player import SamplingEvaluatingPlayerConfig
from ...model import TrainingConfig

from dataclasses import dataclass

@dataclass
class CoachConfig:
    player_config: SamplingEvaluatingPlayerConfig = SamplingEvaluatingPlayerConfig()
    training_config: TrainingConfig = TrainingConfig()
    resume_from_checkpoint: bool = False
    """continue training from previous checkpoint"""
    num_games_per_episode: int = 100
    """number of self-play games per model update"""
    num_iterations: int = 100
    """total number of training iterations"""
    train_buffer_length: int = 20
    """maximum number of games to store in buffer"""
    evaluation_games: int = 20
    """number of games to determine best model"""
    win_threshold: float = .6
    """proportion of wins that a new model must win to be considered the best"""
    save_directory: str = "output"
    """directory to save logs, model, files, etc. to"""
    best_checkpoint_path: str = "model-best"
    """name of best model"""
    successive_win_requirement: int = 7
    """number of games won by best model before training terminates"""
    model_filenames: str = "model-{:06}"
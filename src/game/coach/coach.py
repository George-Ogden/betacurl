from tqdm import trange, tqdm
import numpy as np
import wandb
import os

from typing import List, Optional, Tuple, Type
from copy import copy

from ...sampling import NNSamplingStrategy, RandomSamplingStrategy, SamplingStrategy
from ...evaluation import EvaluationStrategy, NNEvaluationStrategy
from ...utils import SaveableObject

from ..player import Player, SamplingEvaluatingPlayer, SamplingEvaluatingPlayerConfig
from ..game import Game, GameSpec
from ..arena import Arena

from .config import CoachConfig

class Coach(SaveableObject):
    DEFAULT_FILENAME = "coach.pickle"
    SEPARATE_ATTRIBUTES = ["player"]
    def __init__(
        self,
        game: Game,
        SamplingStrategyClass: Type[SamplingStrategy] = NNSamplingStrategy,
        EvaluationStrategyClass: Type[EvaluationStrategy] = NNEvaluationStrategy,
        config: CoachConfig = CoachConfig()
    ):
        self.game = game
        self.setup_player(
            game_spec=game.game_spec,
            SamplingStrategyClass=SamplingStrategyClass,
            EvaluationStrategyClass=EvaluationStrategyClass,
            config=config.player_config
        )

        self.train_example_history = []
        self.set_config(config)

        self.save_directory = config.save_directory
        self.best_model_file = config.best_checkpoint_path
        self.model_filename = config.model_filenames

    def set_config(self, config: CoachConfig):
        self.config = copy(config)
        self.player_config = self.config.player_config
        self.training_config = self.config.training_config

        self.num_iterations = config.num_iterations
        self.num_games_per_episode = config.num_games_per_episode
        self.train_buffer_length = config.train_buffer_length
        self.num_eval_games = config.evaluation_games
        self.win_threshold = int(config.win_threshold * self.num_eval_games)
        self.resume_from_checkpoint = config.resume_from_checkpoint
        assert (self.num_eval_games + 1) // 2 <= self.win_threshold <= self.num_eval_games
        self.learning_patience = config.successive_win_requirement
        # start training with full patience
        self.patience = self.learning_patience

    def setup_player(
        self,
        game_spec: GameSpec,
        SamplingStrategyClass: Type[SamplingStrategy] = NNSamplingStrategy,
        EvaluationStrategyClass: Type[EvaluationStrategy] = NNEvaluationStrategy,
        config: SamplingEvaluatingPlayerConfig = SamplingEvaluatingPlayerConfig()
    ):
        self.player = SamplingEvaluatingPlayer(
            game_spec=game_spec,
            SamplingStrategyClass=SamplingStrategyClass,
            EvaluationStrategyClass=EvaluationStrategyClass,
            config=config
        )

    @property
    def best_checkpoint_path(self) -> str:
        return os.path.join(self.save_directory, self.best_model_file)

    def get_checkpoint_path(self, iteration: int) -> str:
        return os.path.join(self.save_directory, self.model_filename.format(iteration))

    def load_checkpoint(self) -> Optional[int]:
        last_iteration = None
        for i in range(self.num_iterations + 1):
            potential_directory = self.get_checkpoint_path(i)
            if os.path.exists(potential_directory):
                last_iteration = i

        if last_iteration is not None:
            coach = self.load(self.get_checkpoint_path(last_iteration))
            for k, v in vars(coach).items():
                if k != "config":
                    setattr(self, k, v)
                self.set_config(self.config)

            print(f"Successfully loaded model from `{self.get_checkpoint_path(last_iteration)}`")
            return last_iteration

    def learn(self):
        start_iteration = 0
        if self.resume_from_checkpoint:
            start_iteration = self.load_checkpoint() or 0

        print("Starting the learning process")
        self.save_model(current_iteration=start_iteration, wins=-1)

        for iteration in range(start_iteration, self.num_iterations):
            print(f"Starting iteration {iteration}")
            train_arena = Arena([self.best_player.dummy_constructor] * 2, game=self.game)
            train_examples = np.empty(self.num_games_per_episode, dtype=object)
            for i in trange(self.num_games_per_episode, desc="Playing episode"):
                result, game_history = train_arena.play_game(starting_player=i % 2, return_history=True, training=True)
                training_samples = self.transform_history_for_training(game_history)
                train_examples[i] = training_samples
            self.train_example_history.append(train_examples)

            while len(self.train_example_history) > self.train_buffer_length:
                self.train_example_history.pop(0)

            train_examples = [move for histories in self.train_example_history for history in histories for move in history]

            self.player.learn(train_examples, self.game.get_symmetries, self.training_config)

            wins = self.evaluate()
            wandb.log({"best_win_ratio": wins / self.num_eval_games})
            self.save_model(current_iteration=iteration + 1, wins=wins)
            if self.update_patience(wins):
                break

    def update_patience(self, wins: int) -> bool:
        if wins > self.win_threshold:
            self.patience = self.learning_patience
        self.patience -= 1
        return self.patience <= 0

    def benchmark(self, Opponent: Type[Player]) -> Tuple[int, int]:
        eval_arena = Arena([self.player.dummy_constructor, Opponent], game=self.game)
        wins, losses = eval_arena.play_games(self.num_eval_games, display=False, training=False)
        return wins, losses


    def evaluate(self) -> int:
        champion = self.best_player
        wins, losses = self.benchmark(champion.dummy_constructor)
        print(f"Most recent model result: {wins}-{losses} (current-best)")
        return wins

    @property
    def best_player(self):
        if os.path.exists(self.best_checkpoint_path):
            return self.load_player(self.best_checkpoint_path)
        else:
            return SamplingEvaluatingPlayer(
                self.game.game_spec,
                SamplingStrategyClass=RandomSamplingStrategy,
                EvaluationStrategyClass=EvaluationStrategy,
                config=SamplingEvaluatingPlayerConfig(
                    num_train_samples=self.player_config.num_train_samples,
                    num_eval_samples=self.player_config.num_eval_samples,
                    epsilon=self.player_config.epsilon,
                )
            )

    def save_model(self, current_iteration, wins):
        if not os.path.exists(self.save_directory):
            os.mkdir(self.save_directory)

        print(f"Saving model after {current_iteration} learning iteration{'s' * (current_iteration != 1)}")
        self.save(self.get_checkpoint_path(current_iteration))

        if wins > self.win_threshold:
            print("Saving new best model")
            self.save(self.best_checkpoint_path)

    @classmethod
    def load_player(cls, directory: str) -> SamplingEvaluatingPlayer:
        return SamplingEvaluatingPlayer.load(directory)

    @classmethod
    def load(cls, directory: str) -> "Self":
        self = super().load(directory)
        self.player = self.load_player(directory)
        return self

    @staticmethod
    def transform_history_for_training(training_data: List[Tuple[int, np.ndarray, np.ndarray, float]]) -> List[Tuple[int, np.ndarray, np.ndarray, float]]:
        total_reward = 0.
        return [(player, observation, action, total_reward := total_reward + (reward or 0)) for player, observation, action, reward in reversed(training_data)]
from tqdm import trange
import numpy as np
import wandb
import os

from typing import List, Optional, Tuple, Type
from copy import copy

from ..player import Arena, Player, NNMCTSPlayer, NNMCTSPlayerConfig
from ..mcts import Node, Transition
from ..utils import SaveableObject
from ..game import Game, GameSpec

from  .config import CoachConfig

class Coach(SaveableObject):
    DEFAULT_FILENAME = "coach.pickle"
    SEPARATE_ATTRIBUTES = ["player"]
    def __init__(
        self,
        game: Game,
        config: CoachConfig = CoachConfig()
    ):
        self.game = game
        self.setup_player(
            game_spec=game.game_spec,
            config=config.player_config
        )

        self.train_example_history = []
        self.set_config(config)

        self.save_directory = config.save_directory
        self.best_model_file = config.best_checkpoint_path
        self.model_filename = config.model_filenames
        self.stats = {}

    def set_config(self, config: CoachConfig):
        self.config = copy(config)
        self.player_config = self.config.player_config
        self.training_config = self.config.training_config

        self.num_iterations = config.num_iterations
        self.num_games_per_episode = config.num_games_per_episode
        self.num_eval_games = config.evaluation_games
        self.win_threshold = config.win_threshold
        self.resume_from_checkpoint = config.resume_from_checkpoint
        self.eval_simulations = config.num_eval_simulations
        self.learning_patience = config.successive_win_requirement
        # start training with full patience
        self.patience = self.learning_patience

    def setup_player(
        self,
        game_spec: GameSpec,
        config: NNMCTSPlayerConfig = NNMCTSPlayerConfig()
    ):
        self.player = NNMCTSPlayer(
            game_spec=game_spec,
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
        self.save_model(current_iteration=start_iteration)

        for iteration in range(start_iteration, self.num_iterations):
            print(f"Starting iteration {iteration}")
            self.current_best = self.best_player
            train_arena = Arena([self.current_best.dummy_constructor] * self.game.num_players, game=self.game)
            train_examples = np.empty(self.num_games_per_episode, dtype=object)
            for i in trange(self.num_games_per_episode, desc="Playing episode"):
                result, game_history = train_arena.play_game(starting_player=i % self.game.num_players, return_history=True, training=True)
                training_samples = self.transform_history_for_training(game_history)
                train_examples[i] = training_samples
            self.train_example_history.append(train_examples)

            train_examples = [move for histories in self.train_example_history for history in histories for move in history]

            self.player.learn(train_examples, self.game.get_symmetries, self.training_config)

            self.save_model(current_iteration=iteration + 1)
            if self.update():
                break

    def update(self):
        champion = self.best_player
        champion.simulations = self.eval_simulations
        self.current_best = champion
        
        result = self.compare(champion.dummy_constructor)
        if result:
            self.save_best_model()
        self.stats["updated"] = result

        wandb.log(self.stats)
        self.stats = {}
        return self.update_patience(result)
        

    def update_patience(self, change: bool) -> bool:
        if change:
            self.patience = self.learning_patience

        self.patience -= 1
        self.stats["patience"] = self.patience
        return self.patience <= 0

    def compare(self, Opponent: Type[Player]) -> bool:
        """
        Returns:
            bool: True if the current player is better else False
        """
        self.player.simulations = self.eval_simulations

        eval_arena = Arena([self.player.dummy_constructor, Opponent], game=self.game)
        results = np.array(eval_arena.play_games(self.num_eval_games, display=False, training=False))

        self.player.simulations = self.player_config.num_simulations
        # return True if the current player won more than the win threshold
        self.stats |= {
            "win_rate": (results > 0).sum() / len(results),
            "loss_rate": (results < 0).sum() / len(results),
            "draw_rate": (results == 0).sum() / len(results),
            "best_result": results.max(),
            "worst_result": results.min(),
            "avg_result": results.mean(),
        }
        return results.mean() >= (self.win_threshold + 1) / 2

    @property
    def best_player(self) -> NNMCTSPlayer:
        if os.path.exists(self.best_checkpoint_path):
            return self.load_player(self.best_checkpoint_path)
        else:
            return NNMCTSPlayer(
                self.game.game_spec,
                self.config.player_config
            )

    def save_model(self, current_iteration):
        if not os.path.exists(self.save_directory):
            os.mkdir(self.save_directory)

        print(f"Saving model after {current_iteration} learning iteration{'s' * (current_iteration != 1)}")
        self.save(self.get_checkpoint_path(current_iteration))

    def save_best_model(self):
        print("Saving new best model")
        self.save(self.best_checkpoint_path)
        del self.train_example_history[:]

    @classmethod
    def load_player(cls, directory: str) -> NNMCTSPlayer:
        return NNMCTSPlayer.load(directory)

    @classmethod
    def load(cls, directory: str) -> "Self":
        self = super().load(directory)
        self.player = self.load_player(directory)
        return self

    def transform_history_for_training(
            self,
            training_data: List[Tuple[Node, Transition]]
        ) -> List[Tuple[int, np.ndarray, np.ndarray, float, List[Tuple[np.ndarray, float]]]]:
        total_reward = 0.
        mcts = self.current_best.mcts
        mcts.cleanup()
        
        for node, _ in training_data:
            if not node.transitions:
                continue
            
            initial_policy = node.action_probs / node.action_probs.sum()
            enhanced_policy = np.array([transition.num_visits for transition in node.transitions.values()], dtype=float)
            enhanced_policy /= enhanced_policy.sum()
            for transition, initial_prob, enhanced_prob in zip(node.transitions.values(), initial_policy, enhanced_policy):
                # use the policy change to calculate the advantage
                transition.advantage = enhanced_prob - initial_prob

        history = [
            (
                node.game.player_delta,
                node.game.get_observation(),
                transition.action,
                total_reward := (transition.discount or 1.) * total_reward + (transition.reward or 0),
                [
                    (transition.action, transition.advantage)
                    for transition in node.transitions.values()
                ]
            )
            for node, transition in reversed(training_data)
        ]
        return history
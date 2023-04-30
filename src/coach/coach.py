from tqdm import trange
import numpy as np
import os
from copy import deepcopy

from typing import List, Optional, Tuple, Type
from copy import copy

from ..mcts import FixedMCTS, FixedMCTSConfig, MCTSModel, Node, PolicyMCTSModel, Transition
from ..player import Arena, MCTSPlayer, Player, NNMCTSPlayer, NNMCTSPlayerConfig
from ..schedule import GeometricSchedule, LinearSchedule
from ..utils import SaveableObject
from ..game import Game, GameSpec

from  .config import CoachConfig

class Coach(SaveableObject):
    DEFAULT_FILENAME = "coach.pickle"
    SEPARATE_ATTRIBUTES = ["player"]
    def __init__(
        self,
        game: Game,
        config: CoachConfig=CoachConfig(),
        ModelClass: Optional[Type[MCTSModel]]=PolicyMCTSModel
    ):
        self.game = game
        self.ModelClass = ModelClass
        self.player = self.create_player(
            game_spec=game.game_spec,
            config=config.player_config
        )
        self.set_config(config)

        self.save_directory = config.save_directory
        self.last_model_filename = config.last_checkpoint_path
        self.model_filename = config.model_filenames

    def set_config(self, config: CoachConfig):
        self.config = copy(config)
        self.player_config = self.config.player_config
        self.training_config = self.config.training_config
        self.save_directory = self.config.save_directory

        self.num_iterations = config.num_iterations
        self.num_games_per_episode = config.num_games_per_episode
        self.resume_from_checkpoint = config.resume_from_checkpoint
        self.warm_start_games = config.warm_start_games
        self.temperature_schedule = LinearSchedule(
            values=(config.initial_temperature, config.final_temperature),
            range=(0, config.num_iterations)
        )
        self.lr_schedule = GeometricSchedule(
            values=(config.initial_lr, config.final_lr),
            range=(0, config.num_iterations)
        )

    def create_player(
        self,
        game_spec: GameSpec,
        config: NNMCTSPlayerConfig = NNMCTSPlayerConfig()
    ) -> NNMCTSPlayer:
        player = NNMCTSPlayer(
            game_spec=game_spec,
            config=config,
            ModelClass=self.ModelClass
        )
        if player.model is None:
            player.model = player.create_model()
        return player

    def get_checkpoint_path(self, iteration: int) -> str:
        return os.path.join(self.save_directory, self.model_filename.format(iteration))

    def get_last_checkpoint_path(self) -> str:
        return os.path.join(self.save_directory, self.last_model_filename)

    def load_checkpoint(self) -> Optional[int]:
        last_iteration = None
        for i in range(self.num_iterations + 1):
            potential_directory = self.get_checkpoint_path(i)
            if os.path.exists(potential_directory):
                last_iteration = i

        if last_iteration is not None:
            coach = self.load(self.get_checkpoint_path(last_iteration))
            config = self.config
            for k, v in vars(coach).items():
                setattr(self, k, v)
            self.set_config(config)

            print(f"Successfully loaded model from `{self.get_checkpoint_path(last_iteration)}`")
            return last_iteration
    
    def warm_start(self):
        if self.warm_start_games == 0:
            return
        player_config = deepcopy(self.player_config)
        player_config.mcts_config = FixedMCTSConfig(**{
            k: getattr(self.player_config.mcts_config, k)
            for k in FixedMCTSConfig().keys()
        })
        player = MCTSPlayer(
            game_spec=self.game.game_spec,
            MCTSClass=FixedMCTS,
            config=player_config
        )
        num_games = self.num_games_per_episode
        self.num_games_per_episode = self.warm_start_games
        self.run_iteration(iteration=0, player=player)
        self.num_games_per_episode = num_games

    def learn(self):
        start_iteration = None
        if self.resume_from_checkpoint:
            start_iteration = self.load_checkpoint()
        
        if start_iteration is None:
            self.warm_start()
            start_iteration = 0

        print("Starting the learning process")
        self.save_model(current_iteration=start_iteration)

        for iteration in range(start_iteration, self.num_iterations):
            self.run_iteration(iteration + 1)

    def run_iteration(self, iteration: int, player: Optional[Player]=None):
        if player is None:
            player = self.player
        print(f"Starting iteration {iteration}")
        if isinstance(player, NNMCTSPlayer):
            player.fix()
        self.player.default_temperature = self.temperature_schedule[iteration]
        train_arena = Arena([player.dummy_constructor] * self.game.num_players, game=self.game)
        train_examples = np.empty(self.num_games_per_episode, dtype=object)
        for i in trange(self.num_games_per_episode, desc="Playing episode"):
            result, game_history = train_arena.play_game(starting_player=i % self.game.num_players, return_history=True, training=True)
            training_samples = self.transform_history_for_training(game_history)
            train_examples[i] = training_samples

        train_examples = [move for game in train_examples for move in game]

        self.training_config.lr = self.lr_schedule[iteration]
        self.player.learn(train_examples, self.game.get_symmetries, self.training_config)

        self.save_model(current_iteration=iteration)

    def save_model(self, current_iteration: int):
        if not os.path.exists(self.save_directory):
            os.mkdir(self.save_directory)

        print(f"Saving model after {current_iteration} learning iteration{'s' * (current_iteration != 1)}")
        self.save(self.get_checkpoint_path(current_iteration))
        self.save(self.get_last_checkpoint_path())

    def transform_history_for_training(
            self,
            training_data: List[Tuple[Node, Transition]]
        ) -> List[Tuple[int, np.ndarray, np.ndarray, float, List[Tuple[np.ndarray, float]]]]:
        total_reward = 0.
        self.compute_advantages(training_data)

        history = [
            (
                node.game.player_delta,
                node.game.get_observation(),
                transition.action,
                total_reward := (transition.discount or 1.) * total_reward + (transition.reward or 0),
                [
                    (transition.action, transition.advantage, transition.num_visits)
                    for transition in node.transitions.values()
                ]
            )
            for node, transition in reversed(training_data)
        ]
        return history

    def compute_advantages(self, training_data: List[Tuple[Node, Transition]]):
        for node, _ in training_data:
            if not node.transitions:
                continue

            initial_policy = node.action_probs / node.action_probs.sum()
            enhanced_policy = np.array([transition.num_visits for transition in node.transitions.values()], dtype=float)
            # avoid division by zero
            enhanced_policy /= enhanced_policy.sum() or 1.
            for transition, initial_prob, enhanced_prob in zip(node.transitions.values(), initial_policy, enhanced_policy):
                # use the policy change to calculate the advantage
                transition.advantage = enhanced_prob - initial_prob

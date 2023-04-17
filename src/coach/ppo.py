import numpy as np
import wandb

from typing import List, Optional, Tuple, Type

from ..player import Arena, NNMCTSPlayer, NNMCTSPlayerConfig
from ..mcts import Node, PPOMCTSModel, Transition
from ..game import Game, GameSpec

from .single_player import SinglePlayerCoach
from .config import PPOCoachConfig

class PPOCoach(SinglePlayerCoach):
    def __init__(
        self,
        game: Game,
        config: PPOCoachConfig = PPOCoachConfig(),
        ModelClass: Type[PPOMCTSModel] = PPOMCTSModel,
    ):
        super().__init__(game=game, config=config, ModelClass=ModelClass)
        self.eval_environment = game.clone()
        self.best_reward = -float("inf")
    
    def set_config(self, config: PPOCoachConfig):
        super().set_config(config)
        self.gae_lambda = config.gae_lambda

    def update(self) -> float:
        eval_enviroment = self.eval_environment.clone()
        self.player.simulations = self.eval_simulations
        arena = Arena([self.player.dummy_constructor], game=eval_enviroment)
        rewards = arena.play_games(training=False, display=False, num_games=self.num_eval_games)
        reward = np.mean(rewards)

        self.player.simulations = self.player_config.num_simulations

        if reward > self.best_reward:
            self.save_best_model()

        print(f"Most recent model result: {reward:.3f} (avg. reward)")
        wandb.log({"evaluation reward": reward})

        return self.update_patience(reward)

    def create_player(
        self,
        game_spec: GameSpec,
        config: NNMCTSPlayerConfig = NNMCTSPlayerConfig()
    ) -> NNMCTSPlayer:
        player = super().create_player(game_spec, config)
        player.model = player.create_model()
        return player
    
    def load_checkpoint(self) -> Optional[int]:
        iteration = super().load_checkpoint()
        if iteration is not None:
            self.best_reward = -float("inf")
        return iteration

    def update_patience(self, reward: float) -> bool:
        if reward > self.best_reward:
            self.patience = self.learning_patience
        self.patience -= 1
        return self.patience <= 0

    @property
    def best_player(self) -> NNMCTSPlayer:
        return self.player

    def transform_history_for_training(self, training_data: List[Tuple[Node, Transition]]) -> List[Tuple[int, np.ndarray, np.ndarray, float, List[Tuple[np.ndarray, float]]]]:
        self.compute_advantages(training_data)

        # compute td-lambda targets
        value_predictions = self.player.model.predict_values(
            np.array([observation for _, observation, _, _, _ in training_data])
        )
        next_value_predictions = np.concatenate(
            (value_predictions[1:], value_predictions[-1:])
        )
        lambda_reward = value_predictions[-1]
        history = [
            (
                node.game.player_delta,
                node.game.get_observation(),
                transition.action,
                (
                    lambda_reward := (
                        transition.reward or 0.
                    ) + (
                        transition.discount or 1.
                    ) * (
                        self.gae_lambda * lambda_reward
                        + next_value_prediction
                    ) - value_prediction
                ) + value_prediction,
                [
                    (transition.action, transition.advantage)
                    for transition in node.transitions.values()
                ]
            )
            for (node, transition), (value_prediction, next_value_prediction) in reversed(
                list(
                    zip(
                        training_data,
                        zip(value_predictions, next_value_predictions, strict=True),
                        strict=True
                    )
                )
            )
        ]
        return history
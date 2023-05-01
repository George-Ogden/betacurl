import numpy as np

from typing import List, Tuple, Type

from ..mcts import Node, ReinforceMCTSModel, Transition
from ..player import Arena
from ..game import Game

from .config import SinglePlayerCoachConfig
from .coach import Coach

class SinglePlayerCoach(Coach):
    def __init__(
        self,
        game: Game,
        config: SinglePlayerCoachConfig=SinglePlayerCoachConfig(),
        ModelClass: Type[ReinforceMCTSModel]=ReinforceMCTSModel,
    ):
        super().__init__(game=game, config=config, ModelClass=ModelClass)
        self.eval_environment = game.clone()
        self.best_reward = -float("inf")
        assert self.game.num_players == 1, f"the `{type(self).__name__}` class is for single player games only"
    
    def set_config(self, config: SinglePlayerCoachConfig):
        super().set_config(config)
        self.gae_lambda = config.gae_lambda
        self.num_eval_games = config.eval_games
        self.eval_simulations = config.eval_simulations
        self.best_checkpoint_path = config.best_checkpoint_path
    
    def save_model(self, current_iteration: int):
        super().save_model(current_iteration)
        reward = self.evaluate()
        if reward > self.best_reward:
            self.best_reward = reward
            self.save(self.best_checkpoint_path)
    
    def evaluate(self) -> float:
        arena = Arena([self.player.dummy_constructor], game=self.eval_environment.clone())
        self.player.simulations = self.eval_simulations
        reward = np.mean(arena.play_games(self.num_eval_games, display=False, training=False))
        self.player.simulations = self.config.player_config.num_simulations
        return reward
    
    def transform_history_for_training(
        self,
        training_data: List[Tuple[Node, Transition]]
    ) -> List[Tuple[int, np.ndarray, np.ndarray, float, List[Tuple[np.ndarray, float]]]]:
        self.compute_advantages(training_data)

        # compute td-lambda targets
        value_predictions = self.player.model.predict_values(
            np.array([node.game.get_observation() for node, _ in training_data])
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
                    (transition.action, transition.advantage, transition.num_visits)
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
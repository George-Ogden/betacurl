import numpy as np
import wandb

from typing import Optional

from ..player import Arena, NNMCTSPlayer, NNMCTSPlayerConfig
from ..game import Game, GameSpec
from ..mcts import PPOMCTSModel

from .single_player import SinglePlayerCoach
from .config import PPOCoachConfig

class PPOCoach(SinglePlayerCoach):
    def __init__(
        self,
        game: Game,
        config: PPOCoachConfig = PPOCoachConfig()
    ):
        super().__init__(game=game, config=config)
        self.eval_environment = game.clone()
        self.best_reward = -float("inf")

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
        player = NNMCTSPlayer(
            game_spec=game_spec,
            ModelClass=PPOMCTSModel,
            config=config
        )
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
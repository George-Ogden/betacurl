import numpy as np
import os

from typing import List, Tuple

from ..player import NNMCTSPlayer, NNMCTSPlayerConfig
from ..game import Game, GameSpec

from .config import MCTSCoachConfig
from .coach import Coach

class MCTSCoach(Coach):
    def __init__(
        self,
        game: Game,
        config: MCTSCoachConfig = MCTSCoachConfig()
    ):
        super().__init__(
            game=game,
            SamplingStrategyClass=None,
            EvaluationStrategyClass=None,
            config=config
        )
        self.use_intermediate_states = config.use_intermediate_states

    def setup_player(
        self,
        game_spec: GameSpec,
        SamplingStrategyClass = None,
        EvaluationStrategyClass = None,
        config: NNMCTSPlayerConfig = NNMCTSPlayerConfig()
    ):
        self.player = NNMCTSPlayer(
            game_spec=game_spec,
            config=config
        )

    @classmethod
    def load_player(cls, directory: str) -> NNMCTSPlayer:
        return NNMCTSPlayer.load(directory)

    @property
    def best_player(self):
        if os.path.exists(self.best_checkpoint_path):
            best_player = self.load_player(self.best_checkpoint_path)
        else:
            best_player = NNMCTSPlayer(
                self.game.game_spec,
                config=self.config.player_config
            )

        self.current_best = best_player
        return best_player

    def transform_history_for_training(self, training_data: List[Tuple[int, np.ndarray, np.ndarray, float]]) -> List[Tuple[int, np.ndarray, np.ndarray, float]]:
        history = super().transform_history_for_training(training_data)
        if self.use_intermediate_states:
            mcts = self.current_best.mcts
            mcts.freeze()
            history += [
                (node.game.player_deltas[node.game.to_play], node.game.get_observation(), transition.action,  transition.expected_return)
                for node in mcts.nodes.values()
                for transition in node.transitions.values()
            ]
        return history
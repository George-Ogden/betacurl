import numpy as np

from typing import List, Tuple

from ...mcts import SamplingMCTSModel

from ..player import NNMCTSPlayer, NNMCTSPlayerConfig
from ..game import Game, GameSpec

from .config import MCTSCoachConfig
from .mcts import MCTSCoach

class SamplingMCTSCoach(MCTSCoach):
    def __init__(
        self,
        game: Game,
        config: MCTSCoachConfig = MCTSCoachConfig()
    ):
        super().__init__(
            game=game,
            config=config
        )

    def setup_player(
        self,
        game_spec: GameSpec,
        SamplingStrategyClass = None,
        EvaluationStrategyClass = None,
        config: NNMCTSPlayerConfig = NNMCTSPlayerConfig()
    ):
        self.player = NNMCTSPlayer(
            game_spec=game_spec,
            ModelClass=SamplingMCTSModel,
            config=config
        )

    @property
    def best_player(self):
        best_player = super().best_player
        if best_player.ModelClass != SamplingMCTSModel:
            best_player.ModelClass = SamplingMCTSModel
        return best_player
        

    def transform_history_for_training(
        self,
        training_data: List[Tuple[int, np.ndarray, np.ndarray, float]]
    ) -> Tuple[List[Tuple[int, np.ndarray, np.ndarray, float]], List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
        history = super().transform_history_for_training(training_data)
        
        mcts = self.current_best.mcts
        transitions = [
            (node.game.get_observation(), transition.action, mcts.nodes[transition.next_state].game.get_observation())
            for node in mcts.nodes.values()
            for transition in node.transitions.values()
            if not transition.termination
        ]

        return history, transitions
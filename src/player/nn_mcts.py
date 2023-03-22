from typing import Callable, List, Optional, Tuple, Type
import numpy as np

from ..model import Learnable, TrainingConfig
from ..mcts import MCTSModel, NNMCTS
from ..game import Game, GameSpec

from .config import NNMCTSPlayerConfig
from .mcts import MCTSPlayer

class NNMCTSPlayer(MCTSPlayer, Learnable):
    SEPARATE_ATTRIBUTES = ["model"]
    def __init__(
        self,
        game_spec: GameSpec,
        ModelClass: Type[MCTSModel]=MCTSModel,
        config: Optional[NNMCTSPlayerConfig]=NNMCTSPlayerConfig()
    ):
        super().__init__(
            game_spec,
            NNMCTS,
            config=config
        )

        self.scaling_spec = config.scaling_spec
        self.ModelClass = ModelClass
        self.model: Optional[MCTSModel] = None

    def create_mcts(self, game: Game) -> NNMCTS:
        return self.MCTSClass(
            game=game,
            model=self.model,
            config=self.config.mcts_config
        )

    def create_model(self) -> MCTSModel:
        return self.ModelClass(
            game_spec=self.game_spec,
            scaling_spec=self.scaling_spec
        )

    def learn(
        self,
        training_history: List[Tuple[int, np.ndarray, np.ndarray, float, List[Tuple[np.ndarray, float]]]],
        augmentation_function: Callable[[int, np.ndarray, np.ndarray, float], List[Tuple[int, np.ndarray, np.ndarray, float]]],
        training_config: TrainingConfig = TrainingConfig()
    ):
        if self.model is None:
            self.model = self.create_model()
        self.model.learn(training_history, augmentation_function, training_config)
from typing import Callable, List, Optional, Tuple, Type
import numpy as np

from ..mcts import MCTSModel, NNMCTS, NNMCTSMode, ReinforceMCTSModel
from ..model import Learnable, TrainingConfig
from ..game import Game, GameSpec

from .config import NNMCTSPlayerConfig
from .mcts import MCTSPlayer

class NNMCTSPlayer(MCTSPlayer, Learnable):
    SEPARATE_ATTRIBUTES = ["model", "mcts"]
    def __init__(
        self,
        game_spec: GameSpec,
        ModelClass: Optional[Type[MCTSModel]]=None,
        config: Optional[NNMCTSPlayerConfig]=NNMCTSPlayerConfig(),
        initial_mode: NNMCTSMode=NNMCTSMode.WIDENING
    ):
        super().__init__(
            game_spec,
            NNMCTS,
            config=config
        )

        self.ModelClass = ModelClass or ReinforceMCTSModel
        self.model: Optional[MCTSModel] = None
        self.mode = initial_mode

    def create_mcts(self, game: Game) -> NNMCTS:
        return self.MCTSClass(
            game=game,
            model=self.model,
            config=self.config.mcts_config,
            initial_mode=self.mode
        )

    def create_model(self) -> MCTSModel:
        return self.ModelClass(
            game_spec=self.game_spec
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
    
    def set_mode(self, mode: NNMCTSMode):
        self.mode = mode
        self.mcts.set_mode(mode)
    
    def fix(self):
        self.set_mode(NNMCTSMode.FIXED)
    
    def widen(self):
        self.set_mode(NNMCTSMode.WIDENING)
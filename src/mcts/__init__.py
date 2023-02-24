from .config import FixedMCTSConfig, MCTSConfig, MCTSModelConfig, NNMCTSConfig, WideningMCTSConfig
from .widening import WideningMCTS
from .fixed import FixedMCTS
from .nn import MCTSModel, NNMCTS
from .base import MCTS

BEST_MCTS = WideningMCTS
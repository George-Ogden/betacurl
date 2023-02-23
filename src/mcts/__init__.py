from .config import FixedMCTSConfig, MCTSConfig, MCTSModelConfig, WideningMCTSConfig
from .widening import WideningMCTS
from .fixed import FixedMCTS
from .nn import MCTSModel
from .base import MCTS

BEST_MCTS = WideningMCTS
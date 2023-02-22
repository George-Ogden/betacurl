from .config import FixedMCTSConfig, MCTSConfig, WideningMCTSConfig
from .widening import WideningMCTS
from .fixed import FixedMCTS
from .base import MCTS

BEST_MCTS = WideningMCTS
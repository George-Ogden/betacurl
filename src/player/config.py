from dataclasses import dataclass, field
from typing import Union

from ..mcts import MCTSConfig, NNMCTSConfig
from ..utils import Config

@dataclass
class MCTSPlayerConfig(Config):
    mcts_config: Union[MCTSConfig, dict] = field(default_factory=dict)
    num_simulations: int = 50
    """number of fresh moves to simulate in each position"""
    temperature: float = 1.

    def __post_init__(self):
        # one simulation doesn't generate any actions
        assert self.num_simulations >= 2

@dataclass
class NNMCTSPlayerConfig(MCTSPlayerConfig):
    mcts_config: NNMCTSConfig = NNMCTSConfig()
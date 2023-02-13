from ..player import SamplingEvaluatingPlayerConfig, SharedTorsoSamplingEvaluatingPlayer
from ..game import Game, GameSpec

from .config import CoachConfig
from .coach import Coach

class SharedTorsoCoach(Coach):
    def __init__(
        self,
        game: Game,
        config: CoachConfig = CoachConfig()
    ):
        super().__init__(
            game=game,
            SamplingStrategyClass=None,
            EvaluationStrategyClass=None,
            config=config
        )
    
    def setup_player(
        self,
        game_spec: GameSpec,
        SamplingStrategyClass = None,
        EvaluationStrategyClass =  None,
        config: SamplingEvaluatingPlayerConfig = SamplingEvaluatingPlayerConfig()
    ):
        self.player = SharedTorsoSamplingEvaluatingPlayer(
            game_spec=game_spec,
            config=config
        )

    @classmethod
    def load_player(cls, directory: str) -> SharedTorsoSamplingEvaluatingPlayer:
        return SharedTorsoSamplingEvaluatingPlayer.load(directory)
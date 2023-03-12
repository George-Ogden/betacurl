from src.mcts import FixedMCTS, FixedMCTSConfig, WideningMCTS, WideningMCTSConfig
from src.game import Arena, MCTSPlayer, MCTSPlayerConfig
from src.curling import SingleEndCurlingGame

game = SingleEndCurlingGame()
players = [
    MCTSPlayer(
        game.game_spec,
        WideningMCTS,
        config=MCTSPlayerConfig(
            mcts_config=WideningMCTSConfig(
                kappa=.8,
                cpw=1.
            )
        )
    ).dummy_constructor,
    MCTSPlayer(
        game.game_spec,
        FixedMCTS,
        config=MCTSPlayerConfig(
            mcts_config=FixedMCTSConfig(
                num_actions=30
            )
        )
    ).dummy_constructor
]

arena = Arena(players, game)

history = []
for i in range(30):
    result, h = arena.play_game(i % 2, display=True, return_history=True, training=False)
    history.append(h)
    print(result)

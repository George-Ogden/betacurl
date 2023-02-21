import numpy as np
import cv2

from pytest import mark

from src.curling import Curling, SimulationConstants, SingleEndCurlingGame, CURLING_GAME
from src.game import Arena, Game, Player, RandomPlayer
from src.curling.enums import DisplayTime
from src.curling.curling import Canvas

accurate_constants = SimulationConstants(time_intervals=.02)

class ConsistentPlayer(Player):
    def move(self, game: Game) -> np.ndarray:
        return np.array((1.41, 0, 0))

class ConsistentLeftPlayer(Player):
    def move(self, game: Game) -> np.ndarray:
        return np.array((1.41, 5e-2, 0))

class ConsistentRightPlayer(Player):
    def move(self, game: Game) -> np.ndarray:
        return np.array((1.41, -5e-2, 0))

class OutOfBoundsPlayer(Player):
    def move(self, game: Game) -> np.ndarray:
        return np.array((3, 1e-1, 0))

single_end_game = SingleEndCurlingGame()
accurate_game = SingleEndCurlingGame(accurate_constants)
sophisticated_game = CURLING_GAME

random_player = RandomPlayer(single_end_game.game_spec)
good_player = ConsistentPlayer(single_end_game.game_spec)
left_player = ConsistentLeftPlayer(single_end_game.game_spec)
right_player = ConsistentRightPlayer(single_end_game.game_spec)
out_of_bounds_player = OutOfBoundsPlayer(single_end_game.game_spec)


@mark.display
def test_single_game_display():
    # setup
    Curling.num_stones_per_end = 2
    short_game = SingleEndCurlingGame()
    short_arena = Arena([ConsistentLeftPlayer, ConsistentPlayer], short_game)
    Canvas.DISPLAY_TIME = DisplayTime.NO_LAG

    short_arena.play_game(display=True)
    assert cv2.getWindowProperty(Canvas.WINDOW_NAME, cv2.WND_PROP_VISIBLE) != -1

    # cleanup
    cv2.destroyAllWindows()
    Curling.num_stones_per_end = 16

@mark.display
def test_multi_game_display():
    # setup
    Curling.num_stones_per_end = 2
    short_game = SingleEndCurlingGame()
    short_arena = Arena([ConsistentLeftPlayer, ConsistentPlayer], short_game)
    Canvas.DISPLAY_TIME = DisplayTime.NO_LAG

    short_arena.play_games(display=True, num_games=2)
    assert cv2.getWindowProperty(Canvas.WINDOW_NAME, cv2.WND_PROP_VISIBLE) != -1

    # cleanup
    cv2.destroyAllWindows()
    Curling.num_stones_per_end = 16
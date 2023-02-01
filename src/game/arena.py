import numpy as np

from typing import Callable, Iterable, List, Optional, Tuple, Union
from dm_env._environment import StepType
from tqdm import trange

from .game import Game, GameSpec
from .player.base import Player

class Arena():
    def __init__(self, players: Iterable[Callable[[GameSpec], Player]], game: Game) -> None:
        assert len(players) == 2, "only two player games allowed"
        self.players: List[Player] = [Player(game.game_spec) for Player in players]
        self.game = game

    def play_game(self, starting_player: Optional[int] = None, display: bool = False, return_history: bool = False, training: bool = False) -> Union[int, Tuple[int, List[Tuple[int, np.ndarray, np.ndarray, float]]]]:
        """
        Returns:
            Union[int, Tuple[int, List[Tuple[np.ndarray, np.ndarray, float]]]]: return either
                - the final result (return_history=False)
                - a tuple of (reward, history), where history contains tuples of (player_id, observation, action, reward) at each time step
        """
        for player in self.players:
            if training:
                player.train()
            else:
                player.eval()
        time_step = self.game.reset(starting_player=starting_player)
        total_reward = 0
        history = []
        players = self.players
        while time_step.step_type != StepType.LAST:
            player_index = self.game.to_play
            player_delta = self.game.player_delta
            player = players[player_index]
            observation = self.game._get_observation()
            action = player.move(self.game)

            time_step = self.game.step(action, display=display)
            reward = time_step.reward
            if reward is not None:
                total_reward += reward

            if return_history:
                history.append((player_delta, observation, action, reward))

        assert total_reward != 0, "Games cannot end in a draw!"

        if return_history:
            return total_reward, history
        else:
            return total_reward

    def play_games(self, num_games: int, display: bool = False, training: bool = False) -> Tuple[int, int]:
        """
        Returns:
            Tuple[int, int]: (number of games the first player won, number of games the second player won)
        """
        results = [0, 0]
        for i in trange(num_games, desc="Playing games"):
            result = self.play_game(starting_player=i % 2, display=display, return_history=False, training=training)
            results[0 if result > 0 else 1] += 1
        return tuple(results)
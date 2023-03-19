import numpy as np

from typing import Iterable, List, Optional, Tuple, Type, Union
from dm_env import StepType
from tqdm import trange

from ..game.game import Game
from .base import Player

class Arena():
    def __init__(self, players: Iterable[Type[Player]], game: Game) -> None:
        self.players: List[Player] = [Player(game.game_spec) for Player in players]
        self.game = game

    def play_game(
            self,
            starting_player: Optional[int] = None,
            display: bool = False,
            return_history: bool = False,
            training: bool = False
        ) -> Union[int, Tuple[int, List[Tuple[int, np.ndarray, np.ndarray, Optional[float], Optional[float]]]]]:
        """
        Returns:
            Union[int, Tuple[int, List[Tuple[int, np.ndarray, np.ndarray, Optional[float], Optional[float]]]]]: either
                - the final result (return_history=False)
                - a tuple of (reward, history), where history contains tuples of (player_id, observation, action, reward, discount) at each time step
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
                history.append((player_delta, observation, action, reward, time_step.discount))

        assert total_reward != 0, "Games cannot end in a draw!"

        if return_history:
            return total_reward, history
        else:
            return total_reward

    def play_games(self, num_games: int, display: bool = False, training: bool = False) -> Tuple[int, ...]:
        """
        Returns:
            Tuple[int, ...]: (number of games each player won
        """
        results = [0] * self.game.num_players
        for i in trange(num_games, desc="Playing games"):
            result = self.play_game(starting_player=i % self.game.num_players, display=display, return_history=False, training=training)
            results[self.game.player_deltas.index(np.sign(result))] += 1
        return tuple(results)
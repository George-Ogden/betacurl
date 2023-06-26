import numpy as np

from typing import Iterable, List, Optional, Tuple, Type, Union
from tqdm import trange

from ..mcts import Node, Transition
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
        ) -> Union[float, Tuple[float, List[Tuple[Node, Transition]]]]:
        """Returns:
            Union[float, Tuple[float, List[Tuple[int, Node, Transition]]]]: either
            - the final result (return_history=False)
            - a tuple of (reward, history), where history contains tuples of (node, transition) that represents a path through the MCTS
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
        while not time_step.step_type.last():
            player_index = self.game.to_play
            player = players[player_index]
            action = player.move(self.game)
            if return_history:
                node: Node = player.get_current_node(self.game)
                transition: Transition = node.get_transition(action)
                # history.append((node, transition, player.model.generate_distribution(node.game.get_observation())))
                history.append((node, transition))

            time_step = self.game.step(action, display=display)
            reward = time_step.reward
            if reward is not None:
                total_reward += reward

        if return_history:
            return total_reward, history
        else:
            return total_reward

    def play_games(self, num_games: int, display: bool = False, training: bool = False) -> List[float]:
        """
        Returns:
            List[float]: results of each game
        """
        results = [
            self.play_game(starting_player=i % self.game.num_players, display=display, return_history=False, training=training)
            for i in trange(num_games, desc="Playing games")
        ]
        return results
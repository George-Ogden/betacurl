from copy import copy, deepcopy
import numpy as np

from typing import List, Optional, Tuple
from dm_env.specs import BoundedArray
from dm_env import TimeStep

from ..curling.curling import Curling, SimulationConstants, Stone, StoneColor, StoneThrow
from ..game.game import Game, GameSpec

class SingleEndCurlingGame(Game):
    def __init__(self, simulation_constants: SimulationConstants = SimulationConstants()):
        self.curling = Curling()
        assert self.curling.num_stones_per_end % 2 == 0
        self.num_stones_per_end = self.curling.num_stones_per_end
        self.max_round = self.num_stones_per_end
        self.game_spec = GameSpec(
            move_spec=BoundedArray(
                minimum=StoneThrow.bounds[:, 0],
                maximum=StoneThrow.bounds[:, 1],
                dtype=np.float32,
                shape=(len(StoneThrow.bounds), )
            ),
            observation_spec=BoundedArray(
                minimum=(min(StoneColor.RED, StoneColor.YELLOW),) + (0,) * self.num_stones_per_end + (-self.curling.pitch_width / 2, -self.curling.pitch_length / 2) * self.num_stones_per_end + (0,) * self.num_stones_per_end,
                maximum=(max(StoneColor.RED, StoneColor.YELLOW),) + (1,) * self.num_stones_per_end + (self.curling.pitch_width / 2, 0) * self.num_stones_per_end + (1,) * self.num_stones_per_end,
                dtype=np.float32,
                shape = (1 + self.num_stones_per_end + self.num_stones_per_end * 2 + self.num_stones_per_end, )
            )
        )
        self.simulation_constants = simulation_constants
        self.reset()

    def _reset(self) -> TimeStep:
        self.curling.reset(self.stone_to_play)
        self.max_round = self.num_stones_per_end

    def _get_observation(self)-> np.ndarray:
        """
        Returns:
            np.ndarray: concatenation of (stone_to_play (1), round_encoding (max_round), stone_positions (max_round * 2), stone_mask (max_round))
        """
        round_encoding = np.zeros(self.num_stones_per_end, dtype=bool) # number of rounds remaining
        if self.current_round != self.max_round:
            round_encoding[self.max_round - self.current_round - 1] = 1

        metadata = np.concatenate(((self.stone_to_play,), round_encoding)) # next stone, rounds to play
        red_stones = list(filter(lambda stone: stone.color == StoneColor.RED, self.curling.stones))
        yellow_stones = list(filter(lambda stone: stone.color == StoneColor.YELLOW, self.curling.stones))

        red_positions = self.get_stone_positions(red_stones)
        yellow_positions = self.get_stone_positions(yellow_stones)
        position =  np.concatenate((red_positions, yellow_positions)) # positions of stones

        red_mask = self.get_stone_mask(red_stones)
        yellow_mask = self.get_stone_mask(yellow_stones)
        stone_mask = np.concatenate((red_mask, yellow_mask)) # masks where positions are used

        return np.concatenate((metadata, position, stone_mask))

    def get_stone_positions(self, stones: List[Stone]) -> np.ndarray:
        positions = np.concatenate([stone.position for stone in stones]) if len(stones) > 0 else np.zeros(())
        positions.resize(self.num_stones_per_end)
        return positions

    def get_stone_mask(self, stones: List[Stone]) -> np.ndarray:
        stone_mask = np.ones(len(stones))
        stone_mask.resize(self.num_stones_per_end // 2)
        return stone_mask

    def _step(self, action: np.ndarray, display: bool = False) -> Optional[float]:

        if self.in_free_guard_period:
            # record all stones in case of infringement
            stones = copy(self.curling.stones)
            previous_arrangement = deepcopy(self.curling.stones)
            for stone in stones:
                stone.is_guard = self.curling.in_fgz(stone)

        self.curling.throw(
            StoneThrow(
                self.stone_to_play,
                *action
            ),
            display=display,
            constants=self.simulation_constants
        )

        if self.in_free_guard_period:
            for stone in stones:
                # if a stone has been knocked out of play, restore positions
                if stone.is_guard and self.curling.out_of_bounds(stone):
                    self.curling.stones = previous_arrangement
                    break


        if self.current_round == self.max_round - 1:
            # only in house stones count for scoring
            if len(self.curling.stones) == 0 or (score := self.evaluate_position()) == 0:
                # small bonus to the first team to prevent a draw
                return self.eps * -self.stone_to_play
            else:
                return score

    @property
    def in_free_guard_period(self):
        # zero-indexed rounds
        return self.current_round < 5

    def evaluate_position(self) -> int:
        assert len(self.curling.stones) > 0
        return self.curling.evaluate_position()

    @property
    def stone_to_play(self) -> StoneColor:
        return StoneColor._value2member_map_[self.player_delta]

    def get_symmetries(self, player: int, observation: np.ndarray, action: np.ndarray, reward: float) -> List[Tuple[int, np.ndarray, np.ndarray, float]]:
        observation = self.validate_observation(observation)
        action = self.validate_action(action)
        # flip along x
        symmetries = [(player, observation, action, reward), (player, *self.flip_x(observation.copy(), action.copy()), reward)]
        # change who played
        symmetries.extend([self.flip_order(int(player), observation.copy(), action.copy(), float(reward)) for player, observation, action, reward in symmetries])
        return symmetries

    def flip_x(self, observation: np.ndarray, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self.get_positions(observation)[::2] *= -1
        action[[1, 2]] *= -1
        return observation, action

    def get_round_encoding(self, observation: np.ndarray) -> np.ndarray:
        return observation[1:1 + self.num_stones_per_end]

    def get_positions(self, observation: np.ndarray) -> np.ndarray:
        return observation[1 + self.num_stones_per_end:1 + self.num_stones_per_end * 3]

    def get_mask(self, observation: np.ndarray) -> np.ndarray:
        return observation[1 + self.num_stones_per_end * 3:]

    def flip_order(self, player: int, observation: np.ndarray, action: np.ndarray, reward: float) -> Tuple[int ,np.ndarray, np.ndarray, float]:
        observation[0] *= -1

        positions = self.get_positions(observation)
        red_stones = positions[:self.num_stones_per_end].copy()
        yellow_stones = positions[self.num_stones_per_end:].copy()
        positions[:self.num_stones_per_end] = yellow_stones
        positions[self.num_stones_per_end:] = red_stones

        mask = self.get_mask(observation)
        red_mask = mask[:self.num_stones_per_end // 2].copy()
        yellow_mask = mask[self.num_stones_per_end // 2:].copy()
        mask[:self.num_stones_per_end // 2] = yellow_mask
        mask[self.num_stones_per_end // 2:] = red_mask

        return -player, observation, action, -reward

    def get_random_move(self):
        return np.clip(
            np.random.normal(
                loc=StoneThrow.random_parameters[:, 0],
                scale=StoneThrow.random_parameters[:, 1]
            ),
            a_min=StoneThrow.bounds[:,0],
            a_max=StoneThrow.bounds[:,1]
        )
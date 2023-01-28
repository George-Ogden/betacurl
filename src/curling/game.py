from src.curling.curling import Curling, StoneThrow, StoneColor, Stone
from src.game.game import Game, GameSpec

from dm_env._environment import TimeStep
from dm_env.specs import BoundedArray
from typing import List, Optional, Tuple

import numpy as np

class SingleEndCurlingGame(Game):
    def __init__(self):
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
                minimum=(min(StoneColor.RED, StoneColor.YELLOW),) + (0,) * self.num_stones_per_end + (-self.curling.pitch_width / 2, -self.curling.pitch_length) * self.num_stones_per_end + (0,) * self.num_stones_per_end,
                maximum=(max(StoneColor.RED, StoneColor.YELLOW),) + (1,) * self.num_stones_per_end + (self.curling.pitch_width / 2, 0) * self.num_stones_per_end + (1,) * self.num_stones_per_end,
                dtype=np.float32,
                shape = (1 + self.num_stones_per_end + self.num_stones_per_end * 2 + self.num_stones_per_end, )
            )
        )
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
        self.curling.throw(
            StoneThrow(
                self.stone_to_play,
                *action
            ), display=display
        )
        if self.current_round == self.max_round - 1:
            if len(self.curling.stones) == 0:
                # play another two ends
                self.max_round += 2
            else:
                return self.evaluate_position()
    
    def evaluate_position(self) -> int:
        assert len(self.curling.stones) > 0
        return self.curling.evaluate_position()
    
    @property
    def stone_to_play(self) -> StoneColor:
        return StoneColor._value2member_map_[self.player_delta]
    
    def get_symmetries(self, observation: np.ndarray, action: np.ndarray, reward: float) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        # flip along x
        observation = self.validate_observation(observation)
        action = self.validate_action(action)
        symmetries = [(observation, action, reward), (*self.flip_x(observation.copy(), action.copy()), reward)]
        symmetries += [self.flip_order(observation.copy(), action.copy(), float(reward)) for observation, action, reward in symmetries]
        return symmetries
    
    def flip_x(self, observation: np.ndarray, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self.get_positions(observation)[::2] *= -1
        action[[0, 2, 3]] *= -1
        return observation, action
    
    def get_positions(self, observation: np.ndarray) -> np.ndarray:
        return observation[1 + self.num_stones_per_end:1 + self.num_stones_per_end * 3]

    def get_mask(self, observation: np.ndarray) -> np.ndarray:
        return observation[1 + self.num_stones_per_end * 3:]

    def flip_order(self, observation: np.ndarray, action: np.ndarray, reward: float) -> Tuple[np.ndarray, np.ndarray, float]:
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

        return observation, action, -reward
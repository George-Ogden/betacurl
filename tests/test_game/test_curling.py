from dm_env import StepType
import numpy as np

from pytest import mark
import pytest

from src.curling import Curling, SimulationConstants, SingleEndCurlingGame, Stone, StoneColor, CURLING_GAME
from src.game import Arena, Game, Player, RandomPlayer

accurate_constants = SimulationConstants(time_intervals=(.1, .02))

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
        return np.array((2, 1e-1, 0))

single_end_game = SingleEndCurlingGame()
accurate_game = SingleEndCurlingGame(accurate_constants)
sophisticated_game = CURLING_GAME

random_player = RandomPlayer(single_end_game.game_spec)
good_player = ConsistentPlayer(single_end_game.game_spec)
left_player = ConsistentLeftPlayer(single_end_game.game_spec)
right_player = ConsistentRightPlayer(single_end_game.game_spec)
out_of_bounds_player = OutOfBoundsPlayer(single_end_game.game_spec)

random_arena = Arena([RandomPlayer, RandomPlayer], single_end_game)
forced_arena = Arena([ConsistentLeftPlayer, ConsistentPlayer], single_end_game)


def validate_mask(observation: np.ndarray):
    positions = single_end_game.get_positions(observation)
    mask = single_end_game.get_mask(observation)
    assert len(mask) * 2 == len(positions)
    for i in range(len(mask)):
        assert bool(mask[i]) ^ (positions[2*i:2*(i+1)] == 0).all()

def test_game_is_game():
    assert isinstance(sophisticated_game, Game)

def test_correct_number_of_rounds_played():
    assert single_end_game.reset().step_type == StepType.FIRST
    for i in range(15):
        assert single_end_game.step(good_player.move(single_end_game)).step_type == StepType.MID
    assert single_end_game.step(good_player.move(single_end_game)).step_type == StepType.LAST

@mark.slow
def test_to_play_oscillates():
    single_end_game.reset(starting_player=1)
    assert single_end_game.to_play == 1
    expected_stone = single_end_game.stone_to_play
    for i in range(Curling.num_stones_per_end):
        single_end_game.step(good_player.move(single_end_game))
        expected_stone = ~expected_stone
        assert single_end_game.to_play == i % 2
        assert single_end_game.stone_to_play == expected_stone

    single_end_game.reset(starting_player=0)
    assert single_end_game.to_play == 0
    for i in range(Curling.num_stones_per_end):
        single_end_game.step(good_player.move(single_end_game))
        assert single_end_game.to_play == 1 - (i % 2)

def test_valid_actions_are_valid():
    for i in range(1000):
        single_end_game.validate_action(random_player.move(single_end_game))
    pytest.raises(AssertionError, single_end_game.validate_action, np.array((0, 0)))
    pytest.raises(AssertionError, single_end_game.validate_action, np.array((10, 0, 0)))
    pytest.raises(AssertionError, single_end_game.validate_action, np.array((0, 10, 0)))
    pytest.raises(AssertionError, single_end_game.validate_action, np.array((0, 0, 10)))
    pytest.raises(AssertionError, single_end_game.validate_action, np.array((-10, 0, 0)))
    pytest.raises(AssertionError, single_end_game.validate_action, np.array((0, -10, 0)))
    pytest.raises(AssertionError, single_end_game.validate_action, np.array((0, 0, -10)))

def test_valid_observations_are_valid():
    observation = single_end_game.reset().observation
    assert (observation == single_end_game.get_observation()).all()
    single_end_game.validate_observation(observation)
    assert observation.dtype == single_end_game.game_spec.observation_spec.dtype
    validate_mask(observation)
    assert single_end_game.get_observation().dtype == single_end_game.game_spec.observation_spec.dtype
    single_end_game.validate_observation(single_end_game.get_observation())

    observation = single_end_game.step(good_player.move(single_end_game)).observation
    assert (observation == single_end_game.get_observation()).all()
    single_end_game.validate_observation(observation)
    assert observation.dtype == single_end_game.game_spec.observation_spec.dtype

    assert single_end_game.get_observation().dtype == single_end_game.game_spec.observation_spec.dtype
    single_end_game.validate_observation(single_end_game.get_observation())

def test_good_player_always_wins():
    assert forced_arena.play_games(5) == (0, 5)

@mark.probabilistic
def test_random_players_split_wins():
    wins, losses = random_arena.play_games(25)
    assert wins + losses == 25
    assert min(wins, losses) >= 5

def test_observation_is_reasonable():
    single_end_game.reset(1)
    single_end_game.step(right_player.move(single_end_game))
    single_end_game.step(left_player.move(single_end_game))
    first_observation = single_end_game.get_observation()
    single_end_game.step(right_player.move(single_end_game))
    second_observation = single_end_game.get_observation()

    assert (len(first_observation),) == single_end_game.game_spec.observation_spec.shape
    assert (len(second_observation),) == single_end_game.game_spec.observation_spec.shape

    assert first_observation.dtype == single_end_game.game_spec.observation_spec.dtype
    assert second_observation.dtype == single_end_game.game_spec.observation_spec.dtype

    validate_mask(first_observation)
    validate_mask(second_observation)

    assert single_end_game.get_round_encoding(first_observation).sum() == 1
    assert single_end_game.get_round_encoding(second_observation).sum() == 1

    assert ((single_end_game.get_round_encoding(first_observation) == 1) | (single_end_game.get_round_encoding(first_observation) == 0)).all()
    assert ((single_end_game.get_round_encoding(second_observation) == 1) | (single_end_game.get_round_encoding(second_observation) == 0)).all()

    assert np.abs(np.argwhere(single_end_game.get_round_encoding(first_observation)).reshape(()) - np.argwhere(single_end_game.get_round_encoding(second_observation)).reshape(())) == 1

    for stone in single_end_game.curling.stones:
        left_index = np.argwhere(np.abs(second_observation - stone.position[0]) < 1e-6).reshape(())
        right_index = np.argwhere(np.abs(second_observation - stone.position[1]) < 1e-6).reshape(())
        assert left_index + 1 == right_index

    assert single_end_game.get_mask(first_observation).sum() == 2
    assert single_end_game.get_mask(second_observation).sum() == 3

def test_symmetries_are_reasonable():
    single_end_game.reset(1)
    single_end_game.step(right_player.move(single_end_game))
    single_end_game.step(good_player.move(single_end_game))
    single_end_game.step(right_player.move(single_end_game))

    original_observation = single_end_game.get_observation()
    original_action = right_player.move(single_end_game)
    original_reward = 2

    symmetries = single_end_game.get_symmetries(1, original_observation, original_action, original_reward)

    for player, observation, action, reward in symmetries:
        player_delta = player
        validate_mask(observation)

        mask = single_end_game.get_mask(observation)
        original_mask = single_end_game.get_mask(original_observation)

        reward_delta = reward / original_reward
        assert reward_delta == player_delta

        position_delta = np.nan_to_num(original_action[1] / action[1], 1)
        assert np.allclose(original_action[0], action[0]) and np.allclose(original_action[1], action[1] * position_delta) and np.allclose(original_action[2], action[2] * position_delta)

        positions = single_end_game.get_positions(observation)
        original_positions = single_end_game.get_positions(original_observation)
        if player_delta == 1:
            assert (mask == original_mask).all()
            red_stones = positions[:Curling.num_stones_per_end]
            yellow_stones = positions[Curling.num_stones_per_end:]
        else:
            assert (mask[:Curling.num_stones_per_end // 2] == single_end_game.get_mask(original_observation)[Curling.num_stones_per_end // 2:]).all() \
                and (mask[Curling.num_stones_per_end // 2:] == single_end_game.get_mask(original_observation)[:Curling.num_stones_per_end // 2]).all()
            red_stones = positions[Curling.num_stones_per_end:]
            yellow_stones = positions[:Curling.num_stones_per_end]

        for i in range(Curling.num_stones_per_end):
            position = original_positions[2*i:2*i+2]
            if (position == 0).all():
                continue
            if i < Curling.num_stones_per_end / 2:
                left_index = np.argwhere(np.abs(red_stones - position[0] * position_delta) < 1e-6)
                right_index = np.argwhere(np.abs(red_stones - position[1]) < 1e-6).reshape(())
            else:
                left_index = np.argwhere(np.abs(yellow_stones - position[0] * position_delta) < 1e-6)
                right_index = np.argwhere(np.abs(yellow_stones - position[1]) < 1e-6).reshape(())

            assert right_index - 1 in left_index

        assert observation.dtype == single_end_game.game_spec.observation_spec.dtype
        assert action.dtype == single_end_game.game_spec.move_spec.dtype

        assert (single_end_game.get_round_encoding(observation) == single_end_game.get_round_encoding(original_observation)).all()

def test_six_stone_rule_violation():
    accurate_game.reset()
    accurate_game.step(action=np.array((1.35, 0, 0)))
    position = accurate_game.curling.stones[0].position.copy()
    color = accurate_game.curling.stones[0].color
    accurate_game.step(action=np.array((2., 0, 0)))
    assert len(accurate_game.curling.stones) == 1
    assert (accurate_game.curling.stones[0].position == position).all()
    assert accurate_game.curling.stones[0].color == color

def test_six_stone_rule_non_violation():
    accurate_game.reset()
    for i in range(6):
        accurate_game.step(action=np.array((1.35, 0, 0)))
    accurate_game.step(action=np.array((2., 7e-3, 0)))
    assert len(accurate_game.curling.stones) < 6

def test_six_stone_rule_in_house_non_violation():
    accurate_game.reset()
    accurate_game.step(action=np.array((1.41, 0, 0)))
    position = accurate_game.curling.stones[0].position.copy()
    color = accurate_game.curling.stones[0].color
    accurate_game.step(action=np.array((2., 0, 0)))
    assert len(accurate_game.curling.stones) == 0 or\
        ((accurate_game.curling.stones[0].position != position).all() and\
        accurate_game.curling.stones[0].color != color)

def test_six_stone_rule_in_house_non_violation_both_out():
    accurate_game.reset()
    accurate_game.step(action=np.array((1.41, 0, 0)))
    position = accurate_game.curling.stones[0].position.copy()
    color = accurate_game.curling.stones[0].color
    accurate_game.step(action=np.array((2., 0.005, 0)))
    assert len(accurate_game.curling.stones) == 0 or\
        ((accurate_game.curling.stones[0].position != position).all() and\
        accurate_game.curling.stones[0].color != color)

def test_six_stone_rule_non_violation_edge_case():
    accurate_game.reset()
    for i in range(5):
        accurate_game.step(action=np.array((1.35, 0, 0)))
    accurate_game.step(action=np.array((2., 7e-3, 0)))
    assert len(accurate_game.curling.stones) < 5

def test_six_stone_rule_violation_edge_case():
    accurate_game.reset()
    for i in range(4):
        accurate_game.step(action=np.array((1.35, 0, 0)))
    positions = [stone.position.copy() for stone in accurate_game.curling.stones]
    accurate_game.step(action=np.array((2., 7e-3, 0)))
    assert len(accurate_game.curling.stones) == 4
    for position, stone in zip(positions, accurate_game.curling.stones):
        assert (stone.position == position).all()

def test_in_house_evaluation():
    single_end_game.reset()
    single_end_game.step(action=np.array((1.42, 0, 0)))
    assert single_end_game.evaluate_position() == -single_end_game.stone_to_play

def test_out_of_house_evaluation():
    single_end_game.reset()
    single_end_game.curling.stones.append(Stone(StoneColor.RED, position=(0, -Curling.tee_line_position - Curling.target_radii[-1] - Stone.outer_radius * 2)))
    single_end_game.curling.stones.append(Stone(StoneColor.RED, position=(0, -Curling.tee_line_position + Curling.target_radii[-1] + Stone.outer_radius * 2)))
    assert single_end_game.evaluate_position() == 0

def test_drawn_game_where_player_1_starts():
    single_end_game.reset(0)
    starter = single_end_game.stone_to_play
    for i in range(Curling.num_stones_per_end):
        time_step = single_end_game.step(out_of_bounds_player.move(single_end_game))
    assert time_step.step_type == StepType.LAST
    assert np.sign(time_step.reward) == starter

def test_drawn_game_where_player_2_starts():
    single_end_game.reset(1)
    starter = single_end_game.stone_to_play
    for i in range(Curling.num_stones_per_end):
        time_step = single_end_game.step(out_of_bounds_player.move(single_end_game))
    assert time_step.step_type == StepType.LAST
    assert np.sign(time_step.reward) == starter
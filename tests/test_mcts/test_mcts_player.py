from typing import Tuple
import numpy as np

from curling import Curling
from pytest import mark

from src.game import MCTSPlayer, MCTSPlayerConfig, NNMCTSPlayer, NNMCTSPlayerConfig
from src.mcts import FixedMCTS, FixedMCTSConfig, MCTS, NNMCTS, NNMCTSConfig, WideningMCTS, WideningMCTSConfig
from src.game import Arena, Game, RandomPlayer
from src.curling import SingleEndCurlingGame

from tests.utils import StubGame, SparseStubGame

class StubMCTS(MCTS):
    def select_action(self, observation: np.ndarray) -> np.ndarray:
        return self.game.game_spec.move_spec.minimum

    def _get_action_probs(self, game: Game, temperature: float) -> Tuple[np.ndarray, np.ndarray]:
        return np.array([game.game_spec.move_spec.minimum]), np.array((1.,))

stub_game = StubGame()
sparse_stub_game = SparseStubGame(2)

Curling.num_stones_per_end = 2
single_stone_game = SingleEndCurlingGame()
Curling.num_stones_per_end = 16
single_stone_game.num_stones_per_end = 2

player = MCTSPlayer(single_stone_game.game_spec)

arena = Arena([MCTSPlayer, RandomPlayer], single_stone_game)

@mark.probabilistic
def test_training_and_evaluation_matter():
    players = [
        MCTSPlayer(
            single_stone_game.game_spec,
            FixedMCTS,
            config=MCTSPlayerConfig(
                num_simulations=12,
                mcts_config=FixedMCTSConfig(
                    num_actions=10
                )
            )
        )
        for _ in range(2)
    ]

    score = 0
    for i in range(20):
        single_stone_game.reset(i % 2)
        players[0].eval()
        players[1].train()

        assert players[0].temperature == 0
        assert players[1].temperature > 0

        single_stone_game.step(players[single_stone_game.to_play].move(single_stone_game))
        single_stone_game.step(players[single_stone_game.to_play].move(single_stone_game))
        if len(single_stone_game.curling.stones) > 0:
            score += single_stone_game.evaluate_position()
    assert score >= 2

def test_mcts_is_used():
    minimum_mover = MCTSPlayer(
        single_stone_game.game_spec,
        StubMCTS
    )
    move = minimum_mover.move(single_stone_game)
    assert (move == single_stone_game.game_spec.move_spec.minimum).all()

def test_eval_train_are_same_class():
    assert type(player.train()) == type(player)
    assert type(player.eval()) == type(player)

def test_works_with_less_information():
    free_player = MCTSPlayer(
        sparse_stub_game.game_spec,
        config=MCTSPlayerConfig(
            num_simulations=10
        )
    )
    forced_player = MCTSPlayer(
        sparse_stub_game.game_spec,
        config=MCTSPlayerConfig(
            num_simulations=2
        )
    )
    arena = Arena(players=[free_player.dummy_constructor, forced_player.dummy_constructor], game=sparse_stub_game)
    wins, losses = arena.play_games(2)

def test_config_is_used():
    player = MCTSPlayer(
        stub_game.game_spec,
        WideningMCTS,
        config=MCTSPlayerConfig(
            num_simulations=5,
            mcts_config=WideningMCTSConfig(
                cpuct=.15,
                cpw=.8,
                kappa=.2
            )
        )
    )

    stub_game.reset()
    player.move(stub_game)

    assert player.config.num_simulations == 5
    assert player.config.mcts_config.cpuct == .15
    assert player.config.mcts_config.cpw == .8
    assert player.config.mcts_config.kappa == .2

    assert player.simulations == 5
    assert player.mcts.cpuct == .15
    assert player.mcts.cpw == .8
    assert player.mcts.kappa == .2

    assert 4 <= player.mcts.get_node(stub_game.get_observation()).num_visits <= 5

def test_mcts_config_is_used():
    observation_shape = stub_game.game_spec.observation_spec.shape
    player = NNMCTSPlayer(
        stub_game.game_spec,
        config=NNMCTSPlayerConfig(
            num_simulations=5,
            scaling_spec=np.ones(observation_shape),
            mcts_config=NNMCTSConfig(
                max_rollout_depth=4,
                cpuct=3.4
            )
        )
    )

    stub_game.reset()
    player.move(stub_game)

    assert player.config.num_simulations == 5
    assert player.config.mcts_config.cpuct == 3.4

    assert player.simulations == 5
    assert player.mcts.cpuct == 3.4
    assert player.mcts.max_depth == 4
    assert isinstance(player.mcts, NNMCTS)
    assert player.scaling_spec.shape[:len(observation_shape)] == observation_shape
    assert (player.scaling_spec.reshape(-1)[:np.prod(observation_shape)] == 1).all()

    assert 4 <= player.mcts.get_node(stub_game.get_observation()).num_visits <= 5
from copy import deepcopy
import numpy as np

from dm_env.specs import BoundedArray
from typing import Tuple
from pytest import mark

from src.mcts.base import Node, Transition
from src.mcts import FixedMCTS, MCTS
from src.game import Game, GameSpec

from tests.utils import SparseStubGame, StubGame

class MDPStubGame(StubGame):
    def __init__(self, rounds: int = 6):
        self.max_round = rounds
        self.game_spec = GameSpec(
            move_spec=BoundedArray(
                maximum=(self.max_move,) * self.action_size,
                minimum=(0,) * self.action_size,
                dtype=np.float32,
                shape=(self.action_size,),
            ),
            observation_spec=BoundedArray(
                minimum=(-self.max_round // 2 * self.max_move, 0),
                maximum=((self.max_round + 1) // 2 * self.max_move, self.max_round),
                shape=(2,),
                dtype=np.float32,
            ),
        )
        self.reset()

    def _get_observation(self)-> np.ndarray:
        return np.array((self.score[0] - self.score[1], self.current_round))

class MDPSparseStubGame(SparseStubGame):
    def __init__(self, rounds: int = 6):
        self.max_round = rounds
        self.game_spec = GameSpec(
            move_spec=BoundedArray(
                maximum=(self.max_move,) * self.action_size,
                minimum=(0,) * self.action_size,
                dtype=np.float32,
                shape=(self.action_size,),
            ),
            observation_spec=BoundedArray(
                minimum=(-self.max_round // 2 * self.max_move, 0),
                maximum=((self.max_round + 1) // 2 * self.max_move, self.max_round),
                shape=(2,),
                dtype=np.float32,
            ),
        )
        self.reset()

    def _get_observation(self)-> np.ndarray:
        return np.array((self.score[0] - self.score[1], self.current_round))

class MDPStubGameDeterministic(MDPStubGame):
    def get_random_move(self):
        return (self.game_spec.move_spec.minimum + self.game_spec.move_spec.maximum) / 2

game = MDPStubGame(6)
sparse_game = MDPSparseStubGame(6)
deterministic_game = MDPStubGameDeterministic(6)

move_spec = game.game_spec.move_spec

class SimpleMCTS(MCTS):
    def select_action(self, observation: np.ndarray) -> np.ndarray:
        return np.tile(game.max_move / 2, move_spec.shape)

    def _get_action_probs(self, game: Game, temperature: float) -> Tuple[np.ndarray, np.ndarray]:
        return np.array([self.select_action(None)]), np.array([1.])

class RandomMCTS(MCTS):
    def select_action(self, observation: np.ndarray) -> np.ndarray:
        return np.random.rand(*move_spec.shape)

    def _get_action_probs(self, game: Game, temperature: float) -> Tuple[np.ndarray, np.ndarray]:
        actions = self.get_actions(game.get_observation())
        values = np.array([(action.reward + self.nodes[action.next_state].expected_return) * self.game.player_delta for action in actions])
        if temperature == 0:
            probs = np.zeros(len(actions))
            probs[values.argmax()] = 1
        else:
            probs = np.exp(values / temperature)
        return [action.action for action in actions], probs

def test_immutability():
    game.reset()
    first_round = game.current_round
    tree = SimpleMCTS(game)
    tree.search()
    assert game.current_round == first_round

    for i in range(20):
        nodes = deepcopy(tree.nodes)
        tree.search()
        for representation in nodes:
            assert representation in tree.nodes
            assert tree.nodes[representation].game.current_round == nodes[representation].game.current_round
            assert (tree.nodes[representation].game.get_observation() == nodes[representation].game.get_observation()).all()

def test_predetermined_search():
    deterministic_game.reset()
    tree = SimpleMCTS(deterministic_game)
    for i in range(10):
        assert tree.search() == 0.

@mark.probabilistic
def test_search_expectation():
    game.reset()
    tree = RandomMCTS(game)
    searches = np.array([tree.search() for i in range(1000)])
    assert np.abs(searches.mean()) < 10
    assert 4 < searches.std() < 7

@mark.probabilistic
def test_sparse_search_expectation():
    sparse_game.reset()
    tree = RandomMCTS(sparse_game)
    searches = np.array([tree.search() for i in range(1000)])
    assert np.abs(searches.mean()) < 10
    assert 4 < searches.std() < 7

@mark.probabilistic
def test_high_actions_selected():
    game.reset(0)
    tree = RandomMCTS(game)
    
    for i in range(30):
        tree.search(game)
    actions, probs = tree.get_action_probs()
    
    for action in actions:
        game.validate_action(action)
    
    assert np.allclose(probs.sum(), 1)
    assert ((0 <= probs) & (probs <= 1)).all()

    initial_order = np.argsort(np.min(actions, axis=-1))
    final_order = np.argsort(probs)
    n = len(initial_order)
    r_s = 1 - 6 * np.linalg.norm(initial_order - final_order) / (n * (n ** 2 - 1))
    assert r_s > 0.5
    
    game.step(actions[probs.argmax()])
    
    for i in range(30):
        tree.search(game)
    actions, probs = tree.get_action_probs()
    
    assert np.allclose(probs.sum(), 1)
    assert ((0 <= probs) & (probs <= 1)).all()
    
    initial_order = np.argsort(np.min(actions, axis=-1))
    final_order = np.argsort(probs)
    n = len(initial_order)
    r_s = 1 - 6 * np.linalg.norm(initial_order - final_order) / (n * (n ** 2 - 1))
    assert r_s > 0.5
    
    for action in actions:
        game.validate_action(action)

def test_game_persists():
    game = deterministic_game
    game.reset()
    tree = SimpleMCTS(game)
    for i in range(6):
        for _ in range(30):
            tree.search(game)
        actions, probs = tree.get_action_probs(temperature=0)
        if i > 0:
            assert tree.nodes[tree.encode(game.get_observation())].num_visits > 30
        game.step(actions[probs.argmax()])

def test_actions_chosen_by_num_visits():
    game = deterministic_game
    game.reset()
    mcts = FixedMCTS(game)
    mcts.nodes[FixedMCTS.encode(game.get_observation())] = Node(
        transitions={
            FixedMCTS.encode(action): Transition(
                action, termination=True, num_visits=int(11 * action), next_state=None
            ) for action in np.linspace(0., 10., 1)
        },
        game=game
    )
    actions, probs = mcts.get_action_probs(game)
    assert (np.argsort(actions) == np.argsort(probs)).all()

def test_default_move_generator():
    mcts = FixedMCTS(game, 83)
    actions = mcts.generate_moves(game.get_observation())
    assert len(actions) == 83
    for action, probs in actions:
        game.validate_action(action)

def test_specific_action_generator():
    def generate_moves(observation):
        return [(np.repeat(x, game.game_spec.move_spec.shape), x) for x in np.linspace(0., 1., num=4)]
    mcts = FixedMCTS(game, generate_moves)
    actions = mcts.generate_moves(game.get_observation())
    assert len(actions) == 4
    for action, probs in actions:
        game.validate_action(action)

def test_selection_order():
    game.reset()
    def generate_moves(observation):
        return [(np.repeat(x, game.game_spec.move_spec.shape), x) for x in np.linspace(0., 1., num=4)]
    mcts = FixedMCTS(game, generate_moves)
    for i in range(5):
        mcts.search()
        transitions = mcts.get_node(game.get_observation()).transitions
        assert len(transitions) == i
        if i != 4:
            next_action = mcts.select_action(game.get_observation())
            print(next_action)
            for action in transitions.values():
                assert (action.action > next_action).all()
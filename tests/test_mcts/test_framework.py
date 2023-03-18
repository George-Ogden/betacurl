from dm_env import StepType
from copy import deepcopy
import numpy as np

from pytest import mark

from src.mcts import FixedMCTS, FixedMCTSConfig, MCTS, MCTSConfig, WideningMCTS, WideningMCTSConfig
from src.mcts.base import Node, Transition
from src.game import Game

from tests.utils import BinaryStubGame, MDPStubGame, MDPSparseStubGame

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

    def _get_action_probs(self, game: Game, temperature: float):
        return np.array([self.select_action(None)]), np.array([1.])

class RandomMCTS(MCTS):
    def select_action(self, observation: np.ndarray) -> np.ndarray:
        return np.random.rand(*move_spec.shape)

    def _get_action_probs(self, game: Game, temperature: float):
        actions = self.get_actions(game.get_observation())
        values = np.array([(action.reward + self.nodes[action.next_state].expected_return) * self.game.player_delta for action in actions])
        if temperature == 0:
            probs = np.zeros(len(actions))
            probs[values.argmax()] = 1
        else:
            probs = np.exp(values / temperature)
        return [action.action for action in actions], probs

def test_python_dictionaries():
    # make sure dictionary keys stay in order after insertion
    keys = np.random.permutation(20)
    d = {}
    for k in keys:
        d[k] = np.random.random()
    for dict_key, list_key in zip(d, keys):
        assert dict_key == list_key

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

def test_config_is_used():
    # effective use of config is tested in other tests
    mcts = SimpleMCTS(
        game,
        config=MCTSConfig(
            cpuct=2.8
        )
    )
    assert mcts.cpuct == 2.8
    assert mcts.config.cpuct == 2.8

    mcts = FixedMCTS(
        game,
        config=FixedMCTSConfig(
            cpuct=2.9,
            num_actions=4
        )
    )
    assert mcts.cpuct == 2.9
    assert mcts.num_actions == 4

    mcts = WideningMCTS(
        game,
        config=WideningMCTSConfig(
            cpuct=2.4,
            kappa=.4,
            cpw=.3,
        )
    )
    assert mcts.cpuct == 2.4
    assert mcts.kappa == .4
    assert mcts.cpw == .3

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
    mcts = FixedMCTS(
        game,
        config=FixedMCTSConfig(
            num_actions=83
        )
    )
    for i in range(83):
        action, prob = mcts.generate_action(game.get_observation())
        game.validate_action(action)
    assert mcts.num_actions == 83

def test_specific_action_generator():
    moves = [(np.repeat(x, game.game_spec.move_spec.shape), x) for x in np.linspace(0., 1., num=4)]
    def generate_moves(observation):
        return moves.pop(0)
    mcts = FixedMCTS(
        game,
        generate_moves,
        config=FixedMCTSConfig(
            num_actions=4
        )
    )
    for i in range(4):
        action, prob = mcts.generate_action(game.get_observation())
        game.validate_action(action)
    assert len(moves) == 0

def test_selection_order():
    game.reset()
    moves = [(np.repeat(x, game.game_spec.move_spec.shape), x) for x in np.linspace(0., 1., num=4)]
    used_moves = []
    def generate_moves(observation):
        move = moves.pop(0)
        used_moves.append(move)
        return move
    mcts = FixedMCTS(
        game,
        generate_moves,
        config=FixedMCTSConfig(
            num_actions=4
        )
    )
    for i in range(5):
        mcts.search()
        transitions = mcts.get_node(game.get_observation()).transitions
        assert len(transitions) == i
        if i != 4:
            next_action = mcts.select_action(game.get_observation())
            for action in transitions.values():
                assert (action.action > next_action).all()

def test_width_is_accurate():
    game.reset()
    mcts = WideningMCTS(
        game,
        config=WideningMCTSConfig(
            kappa=.9,
            cpw=.4
        )
    )
    for i in range(1000):
        mcts.search(game)
        expected_size = .4 * (i + 1) ** .9
        assert expected_size - 1 <= len(mcts.get_node(game.get_observation()).transitions) <= expected_size + 1

def test_puct_with_no_rewards():
    long_game = MDPSparseStubGame(rounds=50)
    long_game.reset()
    
    mcts = FixedMCTS(
        long_game,
        config=FixedMCTSConfig(
            num_actions=4
        )
    )
    mcts.rollout = (lambda *args, **kwargs: 0.0).__get__(mcts)

    for i in range(20):
        mcts.search()
    root = mcts.get_node(long_game.get_observation())
    actions = root.potential_actions
    
    assert root.num_visits == 20
    assert len(actions) == 4
    assert len(root.transitions) == 4

    for action in actions:
        assert root.get_transition(action).reward == 0.
        assert root.get_transition(action).num_visits == 5 or (
            root.get_transition(action).num_visits == 4
            and (mcts.select_puct_action(long_game.get_observation()) == action).all()
        )

def test_puct_with_policy():
    max_move = MDPStubGame.max_move
    MDPStubGame.max_move = .01
    
    game = MDPStubGame(rounds=4)
    move_spec = game.game_spec.move_spec
    moves = [(move_spec.minimum, 1.), (move_spec.maximum / 100, 0.)]

    mcts = FixedMCTS(
        game,
        config=FixedMCTSConfig(
            num_actions=2
        ),
        action_generator=lambda *args, **kwargs: moves.pop(0) if len(moves) else (np.random.uniform(move_spec.minimum, move_spec.maximum), 1.)
    )

    for i in range(10):
        mcts.search()
    root = mcts.get_node(game.get_observation())
    
    assert root.num_visits == 10
    assert root.get_transition(move_spec.minimum).num_visits >= 8
    assert (not root.get_transition(move_spec.maximum) or
        root.get_transition(move_spec.maximum).num_visits <= 1)

    # cleanup
    MDPStubGame.max_move = max_move

def test_puct_with_rewards():
    game.reset()
    mcts = FixedMCTS(
        game,
        config=FixedMCTSConfig(
            num_actions=10
        ),
    )

    for i in range(100):
        mcts.search()
    
    root = mcts.get_node(game.get_observation())
    values, counts = zip(*[(action.reward + mcts.nodes[action.next_state].expected_return, action.num_visits) for action in root.transitions.values()])

    n = 10
    r_s = 1 - 6 * np.linalg.norm(np.argsort(values) - np.argsort(counts)) / (n * (n ** 2 - 1))
    assert r_s > .5

def test_freezing():
    game = BinaryStubGame()
    game.reset()
    mcts = FixedMCTS(
        game,
        config=FixedMCTSConfig(
            num_actions=10
        ),
    )

    for i in range(101):
        mcts.search()

    mcts.freeze()
    for node in mcts.nodes.values():
        if node.num_visits < 20:
            continue

        advantages = [transition.advantage for transition in node.transitions.values()]
        assert np.allclose(np.mean(advantages), 0.)
        assert np.std(advantages) <= 1. + 1e-6

        n = 10
        r_s = 1 - 6 * np.linalg.norm(np.argsort(advantages) - np.argsort([transition.action.min() for transition in node.transitions.values()])) / (n * (n ** 2 - 1))
        assert r_s > .5

def test_discount_during_mcts():
    game = BinaryStubGame()
    game.discount = .9
    time_step = game.reset()
    mcts = FixedMCTS(
        game,
        config=FixedMCTSConfig(
            num_actions=1
        ),
    )

    previous_reward = None
    while time_step.step_type != StepType.LAST:
        for i in range(1):
            mcts.search(game)
        node = mcts.get_node(game.get_observation())
        if previous_reward is not None:
            assert np.allclose(np.abs(node.expected_return) * .9, previous_reward)
        previous_reward = np.abs(node.expected_return)
        time_step = game.step(game.get_random_move())
        assert time_step.discount == .9
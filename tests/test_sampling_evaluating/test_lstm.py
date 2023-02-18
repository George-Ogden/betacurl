from src.sampling import LSTMSamplingStrategy

from tests.utils import StubGame

game = StubGame()
game_spec = game.game_spec
observation_spec = game_spec.observation_spec
move_spec = game_spec.move_spec

def test_actions_valid():
    strategy = LSTMSamplingStrategy(observation_spec=observation_spec, action_spec=move_spec)
    action = strategy.generate_actions(game.get_observation())
    game.validate_action(action)

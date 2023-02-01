from src.game import SamplingEvaluatingPlayer, SamplingEvaluatingPlayerConfig, RandomPlayer
from src.evaluation import NNEvaluationStrategy, EvaluationStrategy
from src.sampling import NNSamplingStrategy, RandomSamplingStrategy
from src.curling import SingleEndCurlingGame
from src.io import SaveableObject, SaveableModel

from tests.config import cleanup, requires_cleanup, SAVE_DIR
from tests.utils import generic_save_test

game = SingleEndCurlingGame()
observation_spec = game.game_spec.observation_spec
move_spec = game.game_spec.move_spec

def save_load(object: SaveableObject) -> SaveableObject:
    object.save(SAVE_DIR)
    new_object = type(object).load(SAVE_DIR)
    assert type(new_object) == type(object)
    return new_object

@requires_cleanup
def test_saveable_object_saves():
    SaveableObject.DEFAULT_FILENAME = "file"
    object = SaveableObject()
    generic_save_test(object)

    # cleanup
    SaveableObject.DEFAULT_FILENAME = None

@requires_cleanup
def test_saveable_object_saves_and_loads():
    SaveableObject.DEFAULT_FILENAME = "file"
    object1 = SaveableObject()
    object2 = save_load(object1)
    assert object1.DEFAULT_FILENAME == object2.DEFAULT_FILENAME

    # cleanup
    SaveableObject.DEFAULT_FILENAME = None

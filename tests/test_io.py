import numpy as np

from src.model import ModelConfig, DenseModelFactory, BEST_MODEL_FACTORY
from src.io import SaveableMultiModel, SaveableObject

from tests.config import cleanup, requires_cleanup, SAVE_DIR
from tests.utils import generic_save_test

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

@requires_cleanup
def test_saveable_multimodel_saves_and_loads():
    orig_model = SaveableMultiModel()
    orig_model.m1 = BEST_MODEL_FACTORY.create_model(20, 30)
    orig_model.m2 = DenseModelFactory.create_model(
        4,
        8,
        ModelConfig(
            output_activation="tanh"
        )
    )
    orig_model.DEFAULT_FILENAME = "file"
    orig_model.MODELS = {"m1": "m1.h5", "m2": "m2.h5"}
    orig_model.m1(np.random.randn(1, 20))
    orig_model.m2(np.random.randn(1, 4))

    loaded_model = save_load(orig_model)
    assert loaded_model.DEFAULT_FILENAME == orig_model.DEFAULT_FILENAME
    assert list(loaded_model.MODELS.keys()) == list(orig_model.MODELS.keys())
    assert list(loaded_model.MODELS.values()) == list(orig_model.MODELS.values())

    x = np.random.randn(8, 20)
    assert np.allclose(orig_model.m1(x), loaded_model.m1(x))

    x = np.random.randn(8, 4)
    assert np.allclose(orig_model.m2(x), loaded_model.m2(x))
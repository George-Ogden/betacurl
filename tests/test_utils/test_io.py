from tensorflow.keras import Input, Model, Sequential, layers
import numpy as np

from src.model import ModelConfig, DenseModelFactory, BEST_MODEL_FACTORY
from src.utils import SaveableModel, SaveableMultiModel, SaveableObject

from tests.config import cleanup, requires_cleanup, SAVE_DIR
from tests.utils import generic_save_test, save_load

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
def test_saveable_model_saves_and_loads():
    orig_model = SaveableModel()
    
    inputs = Input((7,))
    outputs = layers.Dense(4)(inputs)
    orig_model.model = Model(
        inputs=inputs,
        outputs=outputs
    )

    orig_model.model(np.random.randn(1, 7))
    loaded_model = save_load(orig_model)

    x = np.random.randn(8, 7)
    assert np.allclose(orig_model.model(x), loaded_model.model(x))
    
@requires_cleanup
def test_saveable_model_saves_and_loads_with_no_model():
    orig_model = SaveableModel()
    loaded_model = save_load(orig_model)

    assert orig_model.model is None
    assert loaded_model.model is None

@requires_cleanup
def test_saveable_multimodel_saves_and_loads():
    orig_model = SaveableMultiModel()
    orig_model.m1 = Sequential(
        [
            layers.Rescaling(offset=1., scale=.5),
            BEST_MODEL_FACTORY.create_model(20, 30)
        ]
    )
    orig_model.m2 = DenseModelFactory.create_model(
        4,
        8,
        ModelConfig(
            output_activation="tanh"
        )
    )
    orig_model.m3 = None

    orig_model.DEFAULT_FILENAME = "file"
    orig_model.MODELS = {"m1": "m1.h5", "m2": "m2.h5", "m3": "m3.h5"}
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

    assert loaded_model.model is None
    assert loaded_model.m3 is None

@requires_cleanup
def test_saveable_multimodel_saves_and_loads_with_model():
    orig_model = SaveableMultiModel()
    orig_model.model = DenseModelFactory.create_model(5, 9)
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
    orig_model.model(np.random.randn(1, 5))
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

    x = np.random.randn(6, 5)
    assert np.allclose(orig_model.model(x), loaded_model.model(x))
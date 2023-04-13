import tensorflow as tf
import numpy as np

from pytest import mark

from src.mcts.model.fourier import FourierDistribution
from src.mcts import FourierMCTSModel, FourierMCTSModelConfig
from src.model import TrainingConfig

from tests.utils import MDPStubGame

test_distribution = FourierDistribution(
    coefficients = tf.reshape(tf.range(24, dtype=tf.float32), (4, 3, 2)),
    range = tf.constant([[2., 4.] for _ in range(4)]),
)

max_move = MDPStubGame.max_move
action_size = MDPStubGame.action_size
MDPStubGame.max_move = 1.5
MDPStubGame.action_size = 1
stub_game = MDPStubGame(6)
MDPStubGame.action_size = action_size
MDPStubGame.max_move = max_move

game_spec = stub_game.game_spec
observation_spec = game_spec.observation_spec
move_spec = game_spec.move_spec

move = np.ones(game_spec.move_spec.shape)

result = 3.
training_data = [((-1)**i, np.array((1.25 * ((i + 1) // 2),)), (.25 * move) if i % 2 else (1.25 * move), result, [((.25 * move) if i % 2 else (1.25 * move), (-1.)**i)] * 2) for i in range(6)]
mixed_training_data = training_data + [((-1)**i, np.array((1.25 * ((i + 1) // 2),)), (.25 * move) if i % 2 else (1.25 * move), result, [((.25 * move) if i % 2 else (1.25 * move), (-1.)**i)]) for i in range(6)]
training_data *= 100
mixed_training_data *= 50

def test_distribution_pdf_cdf():
    assert tf.reduce_all(test_distribution.pdf >= 0)
    assert tf.reduce_all(0 <= test_distribution.cdf) and tf.reduce_all(test_distribution.cdf <= 1+1e-2)
    assert np.allclose(test_distribution.cdf[:, 0], 0., atol=1e-2)
    assert np.allclose(test_distribution.cdf[:,-1], 1., atol=1e-2)

def test_distribution_stats():
    mean = test_distribution.mean()
    mode = test_distribution.mode()
    std = test_distribution.stddev()
    variance = test_distribution.variance()

    assert mean.shape in {(4,), (4, 1)}
    assert mode.shape in {(4,), (4, 1)}
    assert std.shape in {(4,), (4, 1)}
    assert variance.shape in {(4,), (4, 1)}

    assert tf.reduce_all(2 <= mean) and tf.reduce_all(mean <= 4)
    assert tf.reduce_all(2 <= mode) and tf.reduce_all(mode <= 4)
    assert tf.reduce_all(0 < variance) and tf.reduce_all(variance < 1)
    assert np.allclose(std, tf.sqrt(variance))

def test_distribution_sample():
    for sample in test_distribution.sample(100):
        assert sample.shape == (4,)
        assert tf.reduce_all(2 <= sample) and tf.reduce_all(sample <= 4)
        assert tf.reduce_all(test_distribution.prob(sample) >= 0)

def test_config_is_used():
    model = FourierMCTSModel(
        game_spec=game_spec,
        config=FourierMCTSModelConfig(
            fourier_features=7,
            feature_size=32
        )
    )

    assert np.prod(model.policy_head(np.random.rand(1, 32)).shape) % 7 == 0

def test_distribution_generation():
    model = FourierMCTSModel(game_spec=game_spec)
    distribution = model.generate_distribution(stub_game.reset().observation)
    assert distribution.sample().shape in {(), (1,)}

@mark.flaky
def test_fourier_model_learns_policy():
    model = FourierMCTSModel(
        game_spec,
        config=FourierMCTSModelConfig(
            vf_coeff=0.,
            ent_coeff=0.,
            fourier_features=4
        ),
    )
    model.learn(training_data, stub_game.get_symmetries)

    assert tf.reduce_all(model.generate_distribution(training_data[0][1]).mean() > move / 2)

@mark.flaky
def test_fourier_model_losses_converge():
    model = FourierMCTSModel(
        game_spec,
        config=FourierMCTSModelConfig(
            vf_coeff=.5,
            fourier_features=4
        ),
    )

    model.learn(
        training_data,
        stub_game.no_symmetries,
        training_config=TrainingConfig(
            training_epochs=10,
            lr=1e-2
        )
    )

    distribution = model.generate_distribution(training_data[0][1])
    assert np.abs(model.predict_values(training_data[0][1]) - result) < stub_game.max_move
    assert tf.reduce_all(distribution.mean() > move / 2)
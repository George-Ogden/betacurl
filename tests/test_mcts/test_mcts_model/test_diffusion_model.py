import numpy as np

from src.mcts import DiffusionMCTSModel, DiffusionMCTSModelConfig

from tests.utils import MDPStubGame

game = MDPStubGame(6)
game_spec = game.game_spec

def test_config_is_used():
    model = DiffusionMCTSModel(
        game_spec=game_spec,
        config=DiffusionMCTSModelConfig(
            diffusion_coef_max=1e-1,
            diffusion_coef_min=1e-5,
            diffusion_steps = 9
        )
    )

    assert (np.argsort(model.betas) == np.arange(9, dtype=int)).all()
    assert np.allclose(model.betas[0], 1e-5)
    assert np.allclose(model.betas[-1], 1e-1)
    for array in (
        model.betas,
        model.noise_weight,
        model.image_weight,
        model.alphas,
        model.posterior_variance,
    ):
        assert np.all(array >= 0) and np.all(array <= 1)
        assert array.dtype == np.float32
        
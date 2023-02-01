from glob import glob
from copy import copy
import numpy as np
import time
import os

from src.sampling import NNSamplingStrategy, RandomSamplingStrategy
from src.evaluation import EvaluationStrategy, NNEvaluationStrategy
from src.model import Learnable, TrainingConfig
from src.game import Coach, CoachConfig

from tests.config import cleanup, requires_cleanup, SAVE_DIR
from tests.utils import StubGame, SparseStubGame

special_cases = dict(
    evaluation_games=4,
    win_threshold=.5,
    successive_win_requirement=4,
)

necessary_config = {
    "save_directory": SAVE_DIR,
}

config_dict = dict(
    resume_from_checkpoint=False,
    num_games_per_episode=2,
    num_iterations=2,
    train_buffer_length=1,
    **necessary_config,
    **special_cases,
    training_config=TrainingConfig(
        training_epochs=10,
        batch_size=64,
        lr=1e-1,
        training_patience=20
    )
)
custom_training_config = copy(config_dict["training_config"])
custom_training_config.training_epochs = 5

stub_game = StubGame(6)
sparse_stub_game = SparseStubGame(6)
observation_spec = stub_game.game_spec.observation_spec
move_spec = stub_game.game_spec.move_spec

boring_coach = Coach(
    game=stub_game,
    SamplingStrategyClass=RandomSamplingStrategy,
    EvaluationStrategyClass=EvaluationStrategy,
    config=CoachConfig(
        **config_dict
    )
)

def test_coach_saves_config():

    assert not os.path.exists(SAVE_DIR)
    for k, v in config_dict.items():
        if k in special_cases:
            continue
        assert getattr(boring_coach, k) == v

    # special cases
    assert boring_coach.win_threshold == 2
    assert boring_coach.num_eval_games == 4
    assert boring_coach.learning_patience == 4

    # assert boring_coach.training_hyperparams["patience"] == 20
    # assert boring_coach.training_hyperparams["epochs"] == 10

    assert type(boring_coach.player.sampler) == RandomSamplingStrategy
    assert type(boring_coach.player.evaluator) == EvaluationStrategy
    assert type(boring_coach.game) == StubGame

class LearningCheck(Learnable):
    times = []
    def learn(self, *args, **kwargs):
        return self.times.append(time.time())
class SampleLearnLogger(RandomSamplingStrategy, LearningCheck):
    ...
class EvaluatorLearnLogger(EvaluationStrategy, LearningCheck):
    ...

@requires_cleanup
def test_coach_uses_config_in_practise():
    coach = Coach(
        game=StubGame(),
        SamplingStrategyClass=SampleLearnLogger,
        EvaluationStrategyClass=EvaluatorLearnLogger,
        config=CoachConfig(
            **config_dict
        )
    )

    coach.player.evaluator_type = EvaluationStrategy
    coach.player.sampler_type = RandomSamplingStrategy

    start_time = time.time()
    coach.learn()
    end_time = time.time()

    assert len(glob(f"{SAVE_DIR}/model-0*")) == 3
    assert len(coach.train_example_history) == 1

    assert len(LearningCheck.times) == 4
    assert (start_time < np.array(LearningCheck.times)).all() and (np.array(LearningCheck.times) < end_time).all()

@requires_cleanup
def test_coach_uses_training_config_with_evaluator():
    coach = Coach(
        game=StubGame(),
        SamplingStrategyClass=RandomSamplingStrategy,
        EvaluationStrategyClass=NNEvaluationStrategy,
        config=CoachConfig(
            **config_dict
        )
    )

    modified_coach = Coach(
        game=StubGame(),
        SamplingStrategyClass=RandomSamplingStrategy,
        EvaluationStrategyClass=NNEvaluationStrategy,
        config=CoachConfig(
            **config_dict |
            {
                "training_config": custom_training_config
            }
        )
    )

    coach.learn()
    modified_coach.learn()

    model = coach.player.evaluator.model
    modified_model = modified_coach.player.evaluator.model
    assert model.history.params["epochs"] == 10
    assert modified_model.history.params["epochs"] == 5
    assert np.allclose(model.optimizer._learning_rate.numpy(), .1)

@requires_cleanup
def test_coach_uses_training_config_with_sampler():
    coach = Coach(
        game=SparseStubGame(),
        SamplingStrategyClass=NNSamplingStrategy,
        EvaluationStrategyClass=EvaluationStrategy,
        config=CoachConfig(
            **config_dict | {
                "num_iterations": 1
            }
        )
    )

    modified_coach = Coach(
        game=SparseStubGame(),
        SamplingStrategyClass=NNSamplingStrategy,
        EvaluationStrategyClass=EvaluationStrategy,
        config=CoachConfig(
            **config_dict |
            {
                "training_config": custom_training_config,
                "num_iterations": 1
            }
        )
    )

    coach.learn()
    modified_coach.learn()

    model = coach.player.sampler.model
    modified_model = modified_coach.player.sampler.model
    assert model._train_counter == 10
    assert modified_model._train_counter == 5
    assert np.allclose(model.optimizer._learning_rate.numpy(), .1)

from copy import copy
from glob import glob
import os

from pytest import mark

from src.game import Arena, Coach, CoachConfig, MCTSCoach, MCTSCoachConfig, NNMCTSPlayer, NNMCTSPlayerConfig, RandomPlayer, SamplingEvaluatingPlayer, SamplingEvaluatingPlayerConfig, SamplingMCTSCoach, SharedTorsoCoach
from src.sampling import RandomSamplingStrategy, SamplingStrategy
from src.evaluation import EvaluationStrategy
from src.mcts import SamplingMCTSModel
from src.model import  TrainingConfig

from src.sampling.range import MaxSamplingStrategy, MinSamplingStrategy
from tests.config import cleanup, requires_cleanup, SAVE_DIR
from tests.utils import BinaryStubGame, StubGame, SparseStubGame

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

def test_reward_transformed_correctly():
    transform = set(Coach.transform_history_for_training(
        [(1, 0, 0, 0), (-1, 10, 10, 1), (1, 20, 20, 2), (-1, 30, 30, 3)],
    ))
    assert transform == set([(1, 0, 0, 6), (-1, 10, 10, 6), (1, 20, 20, 5), (-1, 30, 30, 3)])

def test_reward_transformed_correctly_with_None():
    transform = set(Coach.transform_history_for_training(
        [(1, 0, 0, None), (-1, 10, 10, None), (1, 20, 20, None), (-1, 30, 30, 3)],
    ))
    assert transform == set([(1, 0, 0, 3), (-1, 10, 10, 3), (1, 20, 20, 3), (-1, 30, 30, 3)])

@requires_cleanup
@mark.probabilistic
def test_benchmark():
    player_config = SamplingEvaluatingPlayerConfig(num_eval_samples=10)
    coach = Coach(
        game=stub_game,
        SamplingStrategyClass=RandomSamplingStrategy,
        EvaluationStrategyClass=EvaluationStrategy,
        config=CoachConfig(
            evaluation_games=100,
            resume_from_checkpoint=False,
            **necessary_config,
            player_config=player_config,
        )
    )
    opponent = SamplingEvaluatingPlayer(
        game_spec=stub_game.game_spec,
        SamplingStrategyClass=RandomSamplingStrategy,
        EvaluationStrategyClass=EvaluationStrategy,
        config=player_config
    )
    wins, losses = coach.benchmark(
        opponent.dummy_constructor
    )
    assert min(wins, losses) > 40

class BrokenSamplerCalledException(Exception):
    ...

class BrokenSampler(SamplingStrategy):
    def generate_actions(*args, **kwargs):
        raise BrokenSamplerCalledException()

@requires_cleanup
def test_no_default_best():
    coach = Coach(
        game=stub_game,
        SamplingStrategyClass=BrokenSampler,
        EvaluationStrategyClass=EvaluationStrategy,
        config=CoachConfig(
            num_iterations=0,
            num_games_per_episode=2,
            **necessary_config
        )
    )
    coach.learn()
    assert not os.path.exists(coach.best_checkpoint_path)

@requires_cleanup
def test_warmup():
    coach = Coach(
        game=stub_game,
        SamplingStrategyClass=BrokenSampler,
        EvaluationStrategyClass=EvaluationStrategy,
        config=CoachConfig(
            num_iterations=1,
            num_games_per_episode=2,
            **necessary_config
        )
    )
    try:
        coach.learn()
    except BrokenSamplerCalledException:
        assert type(coach.best_player) != SamplingEvaluatingPlayer or type(coach.best_player.sampler) != BrokenSampler
        assert len(coach.train_example_history[-1]) > 0

@requires_cleanup
def test_sparse_game_for_coaching():
    coach = Coach(
        game=sparse_stub_game,
        SamplingStrategyClass=MaxSamplingStrategy,
        EvaluationStrategyClass=EvaluationStrategy,
        config=CoachConfig(
            num_iterations=1,
            train_buffer_length=1,
            num_games_per_episode=2,
            evaluation_games=10,
            win_threshold=.6,
            **necessary_config
        )
    )
    coach.learn()
    assert type(coach.best_player.sampler) == MaxSamplingStrategy

    coach = Coach(
        game=sparse_stub_game,
        SamplingStrategyClass=MinSamplingStrategy,
        EvaluationStrategyClass=EvaluationStrategy,
        config=CoachConfig(
            num_iterations=1,
            train_buffer_length=1,
            num_games_per_episode=2,
            evaluation_games=10,
            win_threshold=.6,
            **necessary_config
        )
    )
    coach.learn()
    assert type(coach.best_player.sampler) != MinSamplingStrategy

@requires_cleanup
def test_learning_patience():
    coach = Coach(
        game=sparse_stub_game,
        SamplingStrategyClass=MaxSamplingStrategy,
        EvaluationStrategyClass=EvaluationStrategy,
        config=CoachConfig(
            num_iterations=10,
            successive_win_requirement=4,
            train_buffer_length=20,
            num_games_per_episode=2,
            evaluation_games=10,
            win_threshold=.6,
            **necessary_config
        )
    )
    coach.learn()
    assert type(coach.best_player.sampler) == MaxSamplingStrategy
    assert len(glob(f"{SAVE_DIR}/*")) == 6

@requires_cleanup
def test_learning_patience_without_win():
    coach = Coach(
        game=sparse_stub_game,
        SamplingStrategyClass=MinSamplingStrategy,
        EvaluationStrategyClass=EvaluationStrategy,
        config=CoachConfig(
            num_iterations=8,
            successive_win_requirement=4,
            train_buffer_length=20,
            num_games_per_episode=2,
            evaluation_games=10,
            win_threshold=.6,
            **necessary_config
        )
    )
    coach.learn()
    assert type(coach.best_player.sampler) != MinSamplingStrategy
    assert len(glob(f"{SAVE_DIR}/*")) == 5

@requires_cleanup
def test_logs_format(capsys):
    coach = Coach(
        game=stub_game,
        SamplingStrategyClass=RandomSamplingStrategy,
        config=CoachConfig(
            num_iterations=5,
            train_buffer_length=5,
            num_games_per_episode=2,
            evaluation_games=4,
            win_threshold=.6,
            **necessary_config
        )
    )
    coach.learn()

    output = capsys.readouterr()
    assert not "{" in output
    assert not "}" in output

@mark.slow
@mark.probabilistic
@requires_cleanup
def test_shared_model_learns():
    max_move = SparseStubGame.max_move
    SparseStubGame.max_move = 1
    game = SparseStubGame(2)
    coach = SharedTorsoCoach(
        game=game,
        config=CoachConfig(
            save_directory=SAVE_DIR,
            resume_from_checkpoint=False,
            num_games_per_episode=4,
            num_iterations=3,
        )
    )

    coach.learn()

    arena = Arena(game=game, players=[coach.best_player.dummy_constructor, RandomPlayer])
    wins, losses = arena.play_games(10)
    assert wins >= 8

    # cleanup
    SparseStubGame.max_move = max_move

@mark.probabilistic
@mark.slow
@requires_cleanup
def test_mcts_model_learns():
    max_move = BinaryStubGame.max_move
    BinaryStubGame.max_move = 1
    game = BinaryStubGame()
    coach = MCTSCoach(
        game=game,
        config=MCTSCoachConfig(
            resume_from_checkpoint=False,
            num_games_per_episode=100,
            num_iterations=2,
            training_config=TrainingConfig(
                lr=1e-3,
                training_epochs=2
            ),
            player_config=NNMCTSPlayerConfig(
                num_simulations=3
            ),
            use_intermediate_states=False
        )
    )

    coach.learn()

    arena = Arena(
        game=game,
        players=[
            coach.best_player.dummy_constructor, NNMCTSPlayer(
                game_spec=game.game_spec,
                config=NNMCTSPlayerConfig(
                    num_simulations=3
                )
            ).dummy_constructor
        ]
    )
    wins, losses = arena.play_games(100)
    assert wins >= 60

    # cleanup
    BinaryStubGame.max_move = max_move

@mark.probabilistic
@mark.slow
@requires_cleanup
def test_sampling_mcts_model_learns():
    max_move = BinaryStubGame.max_move
    BinaryStubGame.max_move = 1
    game = BinaryStubGame()
    coach = SamplingMCTSCoach(
        game=game,
        config=MCTSCoachConfig(
            **necessary_config,
            resume_from_checkpoint=False,
            num_games_per_episode=100,
            num_iterations=2,
            training_config=TrainingConfig(
                lr=1e-3,
                training_epochs=10
            ),
            player_config=NNMCTSPlayerConfig(
                num_simulations=3
            ),
            use_intermediate_states=False
        )
    )

    coach.learn()

    assert isinstance(coach.player.model, SamplingMCTSModel)
    arena = Arena(
        game=game,
        players=[
            coach.best_player.dummy_constructor, NNMCTSPlayer(
                game_spec=game.game_spec,
                config=NNMCTSPlayerConfig(
                    num_simulations=3
                )
            ).dummy_constructor
        ]
    )
    wins, losses = arena.play_games(100)
    assert wins >= 60

    # cleanup
    BinaryStubGame.max_move = max_move
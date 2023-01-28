from src.game import Arena, Coach, CoachConfig, RandomPlayer, SamplingEvaluatingPlayer, SamplingEvaluatingPlayerConfig
from src.sampling import RandomSamplingStrategy, NNSamplingStrategy, SamplingStrategy
from src.evaluation import EvaluationStrategy

from tests.utils import StubGame, SAVE_DIR, requires_cleanup, cleanup, cleanup_dir, slow
from glob import glob
import numpy as np
import time
import os

special_cases = dict(
    evaluation_games=4,
    win_threshold=.5,
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
    **special_cases
)

stub_game = StubGame(6)
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

def test_coach_saves_config():

    assert not os.path.exists(SAVE_DIR)
    for k, v in config_dict.items():
        if k in special_cases:
            continue
        assert getattr(boring_coach, k) == v
    
    # special cases
    assert boring_coach.win_threshold == 2
    assert boring_coach.num_eval_games == 4

    assert type(boring_coach.player.sampler) == RandomSamplingStrategy
    assert type(boring_coach.player.evaluator) == EvaluationStrategy
    assert type(boring_coach.game) == StubGame

class LearningCheck:
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
def test_checkpoint_restored_correctly():
    coach = Coach(
        stub_game,
        SamplingStrategyClass=RandomSamplingStrategy,
        EvaluationStrategyClass=EvaluationStrategy,
        config=CoachConfig(resume_from_checkpoint=True, **necessary_config)
    )
    coach.player.dummy_variable = 15
    coach.save_model(10, wins=0)
    
    new_coach = Coach(
        stub_game,
        SamplingStrategyClass=RandomSamplingStrategy,
        EvaluationStrategyClass=EvaluationStrategy,
        config=CoachConfig(resume_from_checkpoint=True, **necessary_config)
    )
    iteration = new_coach.load_checkpoint()
    assert iteration == 10
    assert new_coach.player.dummy_variable == 15

@requires_cleanup
def test_checkpoint_restores_in_training():
    coach = Coach(
        stub_game,
        SamplingStrategyClass=RandomSamplingStrategy,
        EvaluationStrategyClass=EvaluationStrategy,
        config=CoachConfig(
            resume_from_checkpoint=True, 
            num_iterations=4, 
            **necessary_config
        )
    )
    coach.dummy_variable = 25
    coach.learn()

    del coach.dummy_variable
    update_time = time.time()
    cleanup_dir(coach.get_checkpoint_path(3))
    cleanup_dir(coach.get_checkpoint_path(4))
    
    coach.learn()

    assert os.path.getmtime(coach.get_checkpoint_path(0)) < update_time
    assert os.path.getmtime(coach.get_checkpoint_path(4)) > update_time
    assert coach.dummy_variable == 25

@requires_cleanup
def test_latent_variable_stored_and_saved():
    coach = Coach(
        game=stub_game, 
        SamplingStrategyClass=NNSamplingStrategy,
        EvaluationStrategyClass=EvaluationStrategy,
        config=CoachConfig(
            player_config=SamplingEvaluatingPlayerConfig(
                latent_size=100
            ),
            **necessary_config
        )
    )
    assert coach.player.sampler.latent_size == 100
    coach.save_model(0, 0)
    del coach.player.sampler.latent_size
    coach.load_checkpoint()
    assert coach.player.sampler.latent_size == 100

@requires_cleanup
def test_training_history_restored():
    coach = Coach(
        game=stub_game, 
        SamplingStrategyClass=RandomSamplingStrategy,
        EvaluationStrategyClass=EvaluationStrategy,
        config=CoachConfig(
            train_buffer_length=25,
            num_iterations=4,
            **necessary_config
        )
    )
    coach.learn()
    del coach.train_example_history
    coach.load_checkpoint()
    assert len(coach.train_example_history) == 4

@requires_cleanup
def test_best_player_saveds_and_loads():
    coach = Coach(
        game=stub_game, 
        SamplingStrategyClass=RandomSamplingStrategy,
        EvaluationStrategyClass=EvaluationStrategy,
        config=CoachConfig(
            train_buffer_length=25,
            num_iterations=1,
            evaluation_games=40,
            **necessary_config
        )
    )
    coach.learn()
    
    champion = SamplingEvaluatingPlayer(stub_game.game_spec)
    champion.dummy_variable = 22
    
    player = coach.player
    coach.player = champion
    coach.save_model(0, 40)
    coach.player = player

    best_player = coach.best_player
    assert best_player.dummy_variable == 22

@requires_cleanup
@slow
def test_model_beats_random_player():
    coach = Coach(
        game=stub_game,
        config=CoachConfig(
            num_games_per_episode=100,
            num_iterations=10,
            **necessary_config
        )
    )
    coach.learn()
    arena = Arena(game=stub_game, players=[coach.best_player.dummy_constructor, RandomPlayer])
    wins, losses = arena.play_games(100, display=True)
    assert wins > 80

    arena = Arena(game=stub_game, players=[coach.best_player.dummy_constructor, coach.load_player(coach.get_checkpoint_path(0))])
    wins, losses = arena.play_games(100, display=True)
    assert wins > 80

@requires_cleanup
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
    
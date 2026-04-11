from src.env import CloudScalerEnv
from src.models import Action
from src.tasks import get_task_medium


def _run_do_nothing_episode(seed: int) -> float:
    env = CloudScalerEnv()
    task, initial, grader = get_task_medium()
    task.seed = seed

    obs = env.reset_for_task(task, initial)
    done = False
    while not done:
        obs, reward, done, info = env.step(Action(action_type="do_nothing"))

    score = grader.grade(env.raw_state())
    return score


def test_deterministic_grader_for_same_seed_and_trajectory() -> None:
    s1 = _run_do_nothing_episode(seed=22)
    s2 = _run_do_nothing_episode(seed=22)
    assert s1 == s2
    assert 0.0 <= s1 <= 1.0

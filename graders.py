from .models import EnvState
from .tasks import get_task_easy, get_task_hard, get_task_medium


def grade_easy(final_state: EnvState) -> float:
    """Grade final state for easy-memory-leak task."""
    return get_task_easy()[2].grade(final_state)


def grade_medium(final_state: EnvState) -> float:
    """Grade final state for medium-traffic-spike task."""
    return get_task_medium()[2].grade(final_state)


def grade_hard(final_state: EnvState) -> float:
    """Grade final state for hard-cascading-failure task."""
    return get_task_hard()[2].grade(final_state)


__all__ = ["grade_easy", "grade_medium", "grade_hard"]

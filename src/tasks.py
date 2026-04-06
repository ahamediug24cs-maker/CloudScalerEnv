from typing import Dict, Tuple

from .models import EnvState, ServiceState, TaskSpec


class TaskGrader:
    def __init__(self, task: TaskSpec):
        self.task = task

    def grade(self, final_state: EnvState) -> float:
        """Deterministic task-specific grade from 0.0 to 1.0."""
        if self.task.difficulty == "easy":
            return self._grade_easy(final_state)
        if self.task.difficulty == "medium":
            return self._grade_medium(final_state)
        return self._grade_hard(final_state)

    def _grade_easy(self, final_state: EnvState) -> float:
        no_crash = 1.0 if final_state.crash_count == 0 else 0.0
        proactive_restart = 1.0 if 1 <= final_state.restart_count <= 4 else 0.2
        budget_score = max(0.0, 1.0 - (final_state.total_budget_used - 45.0) / 70.0)
        score = (0.5 * no_crash) + (0.3 * proactive_restart) + (0.2 * budget_score)
        return round(max(0.0, min(1.0, score)), 3)

    def _grade_medium(self, final_state: EnvState) -> float:
        no_crash = 1.0 if final_state.crash_count == 0 else 0.0
        cpu_control = max(0.0, 1.0 - (final_state.avg_cpu_deviation / 45.0))
        invalid_penalty = max(0.0, 1.0 - (final_state.invalid_action_count / 3.0))
        budget_score = max(0.0, 1.0 - (final_state.total_budget_used - 90.0) / 120.0)
        score = (0.4 * no_crash) + (0.3 * cpu_control) + (0.2 * budget_score) + (0.1 * invalid_penalty)
        return round(max(0.0, min(1.0, score)), 3)

    def _grade_hard(self, final_state: EnvState) -> float:
        no_crash = 1.0 if final_state.crash_count == 0 else 0.0
        cpu_control = max(0.0, 1.0 - (final_state.avg_cpu_deviation / 40.0))
        restart_efficiency = max(0.0, 1.0 - abs(final_state.restart_count - 4) / 6.0)
        budget_score = max(0.0, 1.0 - (final_state.total_budget_used - 180.0) / 160.0)
        invalid_penalty = max(0.0, 1.0 - (final_state.invalid_action_count / 2.0))
        score = (
            (0.35 * no_crash)
            + (0.25 * cpu_control)
            + (0.2 * restart_efficiency)
            + (0.15 * budget_score)
            + (0.05 * invalid_penalty)
        )
        return round(max(0.0, min(1.0, score)), 3)


def get_task_easy() -> Tuple[TaskSpec, Dict[str, ServiceState], TaskGrader]:
    """Task 1: One service with a memory leak; restart before crash."""
    task = TaskSpec(
        task_id="easy-memory-leak",
        name="Memory Leak Prevention",
        difficulty="easy",
        objective="Restart leaking service before it crashes while avoiding unnecessary scaling.",
        max_steps=20,
        seed=11,
    )
    initial = {
        "web-frontend": ServiceState(replicas=2, cpu_utilization=50.0, memory_utilization=85.0),
    }
    return task, initial, TaskGrader(task)


def get_task_medium() -> Tuple[TaskSpec, Dict[str, ServiceState], TaskGrader]:
    """Task 2: CPU traffic spike; scale up dynamically, then scale down."""
    task = TaskSpec(
        task_id="medium-traffic-spike",
        name="Traffic Spike Handling",
        difficulty="medium",
        objective="Keep services in 50-70% CPU band through dynamic scaling and controlled restarts.",
        max_steps=24,
        seed=22,
    )
    initial = {
        "auth-api": ServiceState(replicas=1, cpu_utilization=88.0, memory_utilization=40.0),
        "payment-api": ServiceState(replicas=2, cpu_utilization=60.0, memory_utilization=50.0),
    }
    return task, initial, TaskGrader(task)


def get_task_hard() -> Tuple[TaskSpec, Dict[str, ServiceState], TaskGrader]:
    """Task 3: Multiple services with tighter budget and cascading failures."""
    task = TaskSpec(
        task_id="hard-cascading-failure",
        name="Cascading Failure Mitigation",
        difficulty="hard",
        objective="Prevent cascading crashes under tight budget while maintaining service health.",
        max_steps=30,
        seed=33,
    )
    initial = {
        "frontend": ServiceState(replicas=3, cpu_utilization=80.0, memory_utilization=90.0),
        "backend": ServiceState(replicas=1, cpu_utilization=95.0, memory_utilization=60.0),
        "db-proxy": ServiceState(replicas=2, cpu_utilization=40.0, memory_utilization=40.0),
    }
    return task, initial, TaskGrader(task)


__all__ = ["TaskGrader", "get_task_easy", "get_task_medium", "get_task_hard"]

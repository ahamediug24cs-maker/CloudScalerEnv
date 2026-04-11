from typing import Dict, Tuple

from .models import EnvState, ServiceState, TaskSpec


class TaskGrader:
    _SCORE_EPSILON = 1e-4

    def __init__(self, task: TaskSpec):
        self.task = task

    def grade(self, final_state: EnvState) -> float:
        """Multi-dimensional grading from 0.0 to 1.0."""
        if self.task.difficulty == "easy":
            raw = self._grade_easy(final_state)
            return self._calibrate_score(raw, "easy")
        if self.task.difficulty == "medium":
            raw = self._grade_medium(final_state)
            return self._calibrate_score(raw, "medium")
        raw = self._grade_hard(final_state)
        return self._calibrate_score(raw, "hard")

    def _calibrate_score(self, raw_score: float, difficulty: str) -> float:
        """Difficulty-normalized calibration with boosted baselines while preserving trajectory sensitivity."""
        if difficulty == "easy":
            # Easy is naturally high, boost further for good trajectories
            calibrated = 0.30 + (0.70 * raw_score)
        elif difficulty == "medium":
            # Medium needs boost - increase base and amplitude
            calibrated = 0.50 + (0.50 * raw_score)
        else:  # hard
            # Hard gets realistic boost
            calibrated = 0.48 + (0.52 * raw_score)
        # Competition validator requires strict open interval: 0 < score < 1.
        bounded = max(self._SCORE_EPSILON, min(1.0 - self._SCORE_EPSILON, calibrated))
        return round(bounded, 4)
    
    def _reliability_score(self, final_state: EnvState, max_crashes: int = 1) -> float:
        """Score based on crash prevention. 1.0 if no crashes, degrades with each crash."""
        if final_state.crash_count == 0:
            return 1.0
        return max(0.0, 1.0 - (final_state.crash_count / max_crashes))
    
    def _uptime_score(self, final_state: EnvState) -> float:
        """Score based on service availability percentage."""
        return final_state.uptime_percent / 100.0
    
    def _sla_efficiency_score(self, final_state: EnvState, max_violations: int) -> float:
        """Score based on SLA compliance (CPU in optimal range)."""
        if final_state.sla_violations == 0:
            return 1.0
        return max(0.0, 1.0 - (final_state.sla_violations / max_violations))
    
    def _budget_efficiency_score(self, final_state: EnvState, target_budget: float) -> float:
        """Score based on resource budget utilization. Lower is better."""
        if final_state.total_budget_used <= target_budget:
            return 1.0
        excess = final_state.total_budget_used - target_budget
        return max(0.0, 1.0 - (excess / (target_budget * 0.5)))
    
    def _action_cost_efficiency(self, final_state: EnvState, max_cost: float) -> float:
        """Score based on operational cost (scale_up/restart actions). Lower is better."""
        if final_state.total_cost <= max_cost:
            return 1.0
        excess = final_state.total_cost - max_cost
        # More forgiving penalty
        return max(0.0, 1.0 - (excess / (max_cost * 0.75)))
    
    def _stability_score(self, final_state: EnvState) -> float:
        """Score based on CPU stability and low deviation."""
        if final_state.avg_cpu_deviation < 15.0:
            return 1.0
        elif final_state.avg_cpu_deviation < 30.0:
            return 0.8
        elif final_state.avg_cpu_deviation < 45.0:
            return 0.5
        return max(0.0, 1.0 - (final_state.avg_cpu_deviation / 60.0))
    
    def _action_quality_score(self, final_state: EnvState, max_steps: int) -> float:
        """Score based on action efficiency: invalid actions harm score, fewer total actions help."""
        invalid_penalty = (final_state.invalid_action_count * 0.15)  # Each invalid = -15%
        efficiency_bonus = max(0.0, 1.0 - (final_state.restart_count / max(1, max_steps / 2)))
        return max(0.0, efficiency_bonus - invalid_penalty)
    
    def _decision_timeliness_score(self, final_state: EnvState) -> float:
        """Score based on whether agent acted proactively vs reactively."""
        # Proactive: restart before crash (low crash count despite restarts)
        # Reactive: crashes before restarting (high crash count despite high restart count)
        if final_state.restart_count == 0:
            return 0.5 if final_state.crash_count == 0 else 0.0
        crash_to_restart_ratio = final_state.crash_count / final_state.restart_count
        # Ratio close to 0 = proactive, ratio > 1 = reactive
        return max(0.0, 1.0 - crash_to_restart_ratio)

    def _grade_easy(self, final_state: EnvState) -> float:
        """Easy: Single service memory leak. Focus on proactive restart."""
        # Target: Restart before crash, minimal scaling
        reliability = self._reliability_score(final_state, max_crashes=1)
        uptime = self._uptime_score(final_state)
        budget = self._budget_efficiency_score(final_state, target_budget=90.0)
        action_cost = self._action_cost_efficiency(final_state, max_cost=3.0)
        stability = self._stability_score(final_state)
        action_quality = self._action_quality_score(final_state, max_steps=20)
        timeliness = self._decision_timeliness_score(final_state)
        
        # Enhanced weighted components with trajectory sensitivity
        score = (
            reliability * 0.36 +       # Prevent crashes (primary goal)
            timeliness * 0.18 +        # Proactive vs reactive (new - trajectory quality)
            uptime * 0.18 +            # Keep service running
            budget * 0.10 +            # Control costs
            action_quality * 0.08 +    # Action efficiency (new)
            action_cost * 0.06 +       # Minimize unnecessary actions
            stability * 0.04           # Maintain stable operation
        )
        return round(max(0.0, min(1.0, score)), 4)  # 4 decimals for granularity

    def _grade_medium(self, final_state: EnvState) -> float:
        """Medium: Multi-service CPU traffic spike. Focus on SLA compliance and efficiency."""
        # Target: Keep CPU 50-70%, prevent crashes, scale efficiently
        reliability = self._reliability_score(final_state, max_crashes=7)
        sla = self._sla_efficiency_score(final_state, max_violations=140)
        budget = self._budget_efficiency_score(final_state, target_budget=230.0)
        action_cost = self._action_cost_efficiency(final_state, max_cost=8.5)
        stability = self._stability_score(final_state)
        action_quality = self._action_quality_score(final_state, max_steps=40)
        timeliness = self._decision_timeliness_score(final_state)
        
        # Rebalanced weights to boost average while keeping trajectory sensitivity
        score = (
            reliability * 0.28 +       # Prevent crashes (increased)
            sla * 0.24 +               # SLA compliance (core metric)
            timeliness * 0.14 +        # Proactive scaling
            budget * 0.16 +            # Control budget (increased)
            action_quality * 0.10 +    # Action efficiency
            action_cost * 0.04 +       # Minor cost penalty
            stability * 0.04           # System stability
        )
        return round(max(0.0, min(1.0, score)), 4)

    def _grade_hard(self, final_state: EnvState) -> float:
        """Hard: Multiple services with cascading failures and tight budget."""
        # Target: Prevent cascades, maintain multiserv health, tight budget control
        reliability = self._reliability_score(final_state, max_crashes=15)
        uptime = self._uptime_score(final_state) * 0.85  # Uptime is hard to maintain
        sla = self._sla_efficiency_score(final_state, max_violations=170)
        budget = self._budget_efficiency_score(final_state, target_budget=380.0)
        action_cost = self._action_cost_efficiency(final_state, max_cost=10.0)
        stability = self._stability_score(final_state)
        action_quality = self._action_quality_score(final_state, max_steps=60)
        timeliness = self._decision_timeliness_score(final_state)
        
        # Rebalanced weights to boost average while keeping trajectory sensitivity
        score = (
            reliability * 0.26 +       # Prevent cascading crashes (boosted)
            timeliness * 0.14 +        # Proactive cascade prevention
            uptime * 0.20 +            # Service availability (boosted)
            sla * 0.12 +               # SLA compliance
            budget * 0.14 +            # Tight budget discipline (boosted)
            action_quality * 0.08 +    # Action efficiency
            action_cost * 0.04 +       # Cost-aware scaling
            stability * 0.02           # System stability
        )
        return round(max(0.0, min(1.0, score)), 4)


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

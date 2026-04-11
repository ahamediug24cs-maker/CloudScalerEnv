from __future__ import annotations

from typing import List

from .models import Action, Observation


def _critical_order(task_name: str) -> List[str]:
    if task_name == "hard-cascading-failure":
        return ["db-proxy", "backend", "frontend"]
    if task_name == "medium-traffic-spike":
        return ["auth-api", "payment-api"]
    return ["web-frontend"]


def _predict_cpu_trend(current_cpu: float, step: int, max_steps: int) -> float:
    """Predict CPU trend: ascending if early, descending if late."""
    progress = step / max_steps
    if progress < 0.3:
        return current_cpu * 1.15  # Predict rise in early phase
    elif progress > 0.7:
        return current_cpu * 0.85  # Predict decline in late phase
    return current_cpu


def _easy_policy(obs: Observation) -> Action:
    upper_cpu = 75.0
    lower_cpu = 45.0
    late_horizon = obs.step_count >= int(0.55 * obs.max_steps)
    over_budget = obs.total_budget_used > 60.0

    svc = obs.services["web-frontend"]
    
    # Proactive prediction-based scaling
    predicted_cpu = _predict_cpu_trend(svc.cpu_utilization, obs.step_count, obs.max_steps)

    if svc.status == "crashed":
        return Action(action_type="restart", service_name="web-frontend")
    if svc.memory_utilization >= 90.0:
        return Action(action_type="restart", service_name="web-frontend")
    if svc.status == "degraded" and svc.memory_utilization >= 84.0:
        return Action(action_type="restart", service_name="web-frontend")
    # Predict ahead for CPU scaling
    if predicted_cpu > upper_cpu and svc.replicas < 10:
        return Action(action_type="scale_up", service_name="web-frontend", count=1)
    if (late_horizon or over_budget) and svc.replicas > 1 and svc.cpu_utilization < (lower_cpu - 3.0):
        return Action(action_type="scale_down", service_name="web-frontend", count=1)
    if svc.replicas > 1 and svc.cpu_utilization < lower_cpu and svc.status != "crashed":
        return Action(action_type="scale_down", service_name="web-frontend", count=1)
    return Action(action_type="do_nothing")


def _medium_policy(obs: Observation) -> Action:
    upper_cpu = 80.0
    lower_cpu = 52.0
    late_horizon = obs.step_count >= int(0.50 * obs.max_steps)
    services = obs.services
    auth = services["auth-api"]
    payment = services["payment-api"]
    
    # SLA-aware thresholds based on violation count
    sla_buffer = (obs.sla_violations / 140.0) * 5.0  # Increase threshold if SLA degrading
    adaptive_upper = upper_cpu + min(sla_buffer, 8.0)

    if auth.status == "crashed" or auth.memory_utilization >= 92.0:
        return Action(action_type="restart", service_name="auth-api")
    if payment.status == "crashed" or payment.memory_utilization >= 92.0:
        return Action(action_type="restart", service_name="payment-api")
    # Adaptive SLA-aware scaling
    if auth.cpu_utilization > adaptive_upper and auth.replicas < 3:
        return Action(action_type="scale_up", service_name="auth-api", count=1)
    if payment.cpu_utilization > adaptive_upper and payment.replicas < 2:
        return Action(action_type="scale_up", service_name="payment-api", count=1)
    if late_horizon:
        if payment.cpu_utilization < lower_cpu and payment.replicas > 1:
            return Action(action_type="scale_down", service_name="payment-api", count=1)
        if auth.cpu_utilization < lower_cpu and auth.replicas > 1:
            return Action(action_type="scale_down", service_name="auth-api", count=1)
    return Action(action_type="do_nothing")


def _hard_policy(obs: Observation) -> Action:
    upper_cpu = 76.0
    lower_cpu = 52.0
    late_horizon = obs.step_count >= int(0.65 * obs.max_steps)
    services = obs.services
    db_proxy = services["db-proxy"]
    backend = services["backend"]
    frontend = services["frontend"]
    
    # Adaptive thresholds based on crash history (proactive cascade prevention)
    crash_factor = (obs.crash_count / 15.0) * 3.0
    adaptive_upper = upper_cpu - min(crash_factor, 5.0)  # Lower threshold if crashes increasing

    if backend.status == "crashed" or backend.memory_utilization >= 85.0:
        return Action(action_type="restart", service_name="backend")
    if db_proxy.status == "crashed" or db_proxy.memory_utilization >= 85.0:
        return Action(action_type="restart", service_name="db-proxy")
    if frontend.status == "crashed" or frontend.memory_utilization >= 88.0:
        return Action(action_type="restart", service_name="frontend")
    # Cascading prevention: scale backend first to reduce DB load
    if backend.cpu_utilization > adaptive_upper and backend.replicas < 5:
        return Action(action_type="scale_up", service_name="backend", count=1)
    if db_proxy.cpu_utilization > adaptive_upper and db_proxy.replicas < 4:
        return Action(action_type="scale_up", service_name="db-proxy", count=1)
    if frontend.cpu_utilization > upper_cpu and frontend.replicas < 4:
        return Action(action_type="scale_up", service_name="frontend", count=1)
    if late_horizon:
        if frontend.cpu_utilization < lower_cpu and frontend.replicas > 1:
            return Action(action_type="scale_down", service_name="frontend", count=1)
        if backend.cpu_utilization < lower_cpu and backend.replicas > 1:
            return Action(action_type="scale_down", service_name="backend", count=1)
        if db_proxy.cpu_utilization < lower_cpu and db_proxy.replicas > 1:
            return Action(action_type="scale_down", service_name="db-proxy", count=1)
    return Action(action_type="do_nothing")


def choose_action(obs: Observation, task_name: str) -> Action:
    if task_name == "easy-memory-leak":
        return _easy_policy(obs)
    if task_name == "medium-traffic-spike":
        return _medium_policy(obs)
    if task_name == "hard-cascading-failure":
        return _hard_policy(obs)

    # Safe fallback for unknown custom tasks.
    for name in _critical_order(task_name):
        svc = obs.services.get(name)
        if svc is None:
            continue
        if svc.status == "crashed" or svc.memory_utilization >= 90.0:
            return Action(action_type="restart", service_name=name)
        if svc.cpu_utilization > 75.0 and svc.replicas < 10:
            return Action(action_type="scale_up", service_name=name, count=1)
    return Action(action_type="do_nothing")

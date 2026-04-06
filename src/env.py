import random
from typing import Any, Dict, Optional, Tuple

from .models import Action, EnvState, Observation, Reward, ServiceState, TaskSpec


class CloudScalerEnv:
    def __init__(self):
        self._state: Optional[EnvState] = None
        self._target_cpu = 60.0  # Optimal CPU utilization target.
        self._rng = random.Random(0)

    def reset(
        self,
        initial_services: Dict[str, ServiceState],
        max_steps: int = 20,
        task_id: str = "custom",
        seed: int = 0,
    ) -> Observation:
        """Reset the environment with a specific task configuration."""
        self._rng = random.Random(seed)
        services_copy = {name: service.model_copy(deep=True) for name, service in initial_services.items()}
        self._state = EnvState(
            step_count=0,
            max_steps=max_steps,
            services=services_copy,
            total_budget_used=sum(service.replicas for service in services_copy.values()),
            task_id=task_id,
            seed=seed,
            crash_count=0,
            restart_count=0,
            invalid_action_count=0,
            avg_cpu_deviation=0.0,
            cumulative_reward=0.0,
        )
        return self.state()

    def reset_for_task(self, task: TaskSpec, initial_services: Dict[str, ServiceState]) -> Observation:
        return self.reset(
            initial_services=initial_services,
            max_steps=task.max_steps,
            task_id=task.task_id,
            seed=task.seed,
        )

    def state(self) -> Observation:
        """Return the current strongly-typed state."""
        if self._state is None:
            raise RuntimeError("Environment is not initialized. Call reset() before state().")
        return Observation.model_validate(self._state.model_dump())

    def raw_state(self) -> EnvState:
        """Return the full internal state for deterministic grading and debugging."""
        if self._state is None:
            raise RuntimeError("Environment is not initialized. Call reset() before raw_state().")
        return self._state.model_copy(deep=True)

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """Execute an action, advance the environment, and calculate rewards."""
        if self._state is None:
            raise RuntimeError("Environment is not initialized. Call reset() before step().")
        current_state = self._state

        if current_state.step_count >= current_state.max_steps:
            terminal_reward = Reward(value=0.0)
            return self.state(), terminal_reward, True, {"msg": "Max steps reached."}

        # 1. Apply action to the targeted service.
        invalid_action = False
        if action.action_type != "do_nothing" and action.service_name in current_state.services:
            svc = current_state.services[action.service_name]
            count = action.count or 1

            if action.action_type == "scale_up":
                svc.replicas = min(10, svc.replicas + count)
                # CPU drops as load gets distributed across more replicas.
                svc.cpu_utilization = max(10.0, svc.cpu_utilization - (15.0 * count))
            elif action.action_type == "scale_down":
                svc.replicas = max(1, svc.replicas - count)
                # CPU rises when load concentrates on fewer replicas.
                svc.cpu_utilization = min(100.0, svc.cpu_utilization + (25.0 * count))
            elif action.action_type == "restart":
                svc.status = "healthy"
                svc.memory_utilization = 20.0  # Restart clears leak accumulation.
                current_state.restart_count += 1
        elif action.action_type != "do_nothing":
            invalid_action = True
            current_state.invalid_action_count += 1

        # 2. Simulate environment dynamics: traffic spikes and memory leaks.
        for svc in current_state.services.values():
            if svc.status != "crashed":
                # Simulated traffic fluctuations (-5% to +15%).
                svc.cpu_utilization = min(100.0, max(0.0, svc.cpu_utilization + self._rng.uniform(-5, 15)))
                # Simulated memory leak (+2% to +5% per step).
                svc.memory_utilization = min(100.0, svc.memory_utilization + self._rng.uniform(2, 5))

                if 85.0 <= svc.memory_utilization <= 95.0:
                    svc.status = "degraded"
                elif svc.status == "degraded" and svc.memory_utilization < 85.0:
                    svc.status = "healthy"

                if svc.memory_utilization > 95.0 or svc.cpu_utilization >= 100.0:
                    if svc.status != "crashed":
                        current_state.crash_count += 1
                    svc.status = "crashed"
                    svc.cpu_utilization = 0.0

        current_state.step_count += 1
        current_state.total_budget_used += sum(service.replicas for service in current_state.services.values())
        cpu_deviation = sum(abs(service.cpu_utilization - self._target_cpu) for service in current_state.services.values())
        current_state._cpu_deviation_sum += cpu_deviation
        denom = current_state.step_count * max(1, len(current_state.services))
        current_state.avg_cpu_deviation = round(current_state._cpu_deviation_sum / denom, 4)

        done = current_state.step_count >= current_state.max_steps
        reward = self._calculate_reward(invalid_action=invalid_action)
        current_state.cumulative_reward += reward.value

        info = {
            "task_id": current_state.task_id,
            "seed": current_state.seed,
            "step": current_state.step_count,
            "crashes": current_state.crash_count,
            "invalid_actions": current_state.invalid_action_count,
        }
        return self.state(), reward, done, info

    def _calculate_reward(self, invalid_action: bool) -> Reward:
        """Dense reward signal balancing health, utilization, and budget discipline."""
        if self._state is None:
            raise RuntimeError("Environment is not initialized.")
        state = self._state

        cpu_balance = 0.0
        memory_safety = 0.0
        reliability = 0.0
        budget_efficiency = 0.0

        for svc in state.services.values():
            if svc.status == "crashed":
                reliability -= 1.0
            else:
                cpu_diff = abs(svc.cpu_utilization - self._target_cpu)
                cpu_balance += max(0.0, 1.0 - (cpu_diff / 40.0))
                memory_safety += max(0.0, 1.0 - (svc.memory_utilization / 100.0))
                reliability += 0.2 if svc.status == "healthy" else 0.0
                budget_efficiency += max(0.0, 1.0 - (svc.replicas - 1) / 9.0)

        invalid_penalty = -0.5 if invalid_action else 0.0
        total = cpu_balance + memory_safety + reliability + budget_efficiency + invalid_penalty
        return Reward(
            value=round(total, 4),
            cpu_balance=round(cpu_balance, 4),
            memory_safety=round(memory_safety, 4),
            reliability=round(reliability, 4),
            budget_efficiency=round(budget_efficiency, 4),
            invalid_action_penalty=invalid_penalty,
        )

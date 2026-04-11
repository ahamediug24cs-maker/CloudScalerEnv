import random
from typing import Any, Dict, Optional, Tuple

from .models import Action, EnvState, Observation, Reward, ServiceState, TaskSpec


class CloudScalerEnv:
    def __init__(self):
        self._state: Optional[EnvState] = None
        self._target_cpu = 60.0  # Optimal CPU utilization target.
        self._rng = random.Random(0)
        # Define service dependencies for different tasks
        self._task_dependencies = {
            "easy-memory-leak": {},
            "medium-traffic-spike": {"auth-api": [], "payment-api": ["auth-api"]},  # payment depends on auth
            "hard-cascading-failure": {"frontend": ["backend"], "backend": ["db-proxy"], "db-proxy": []},
        }

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
        
        # Set up service dependencies for this task (default to empty if not recognized)
        dependencies = self._task_dependencies.get(task_id, {})
        
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
            sla_violations=0,
            total_cost=0.0,
            uptime_percent=100.0,
            service_dependencies=dependencies,
            action_costs=0.0,
            cpu_deviation_sum=0.0,
            service_uptime={name: 0 for name in services_copy.keys()},
        )
        return self.state()

    def reset_for_task(self, task: TaskSpec, initial_services: Dict[str, ServiceState]) -> Observation:
        return self.reset(
            initial_services=initial_services,
            max_steps=task.max_steps,
            task_id=task.task_id,
            seed=task.seed,
        )

    def close(self) -> None:
        """No-op close hook for API parity with evaluator runtimes."""
        return None

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
        action_cost = 0.0
        if action.action_type != "do_nothing" and action.service_name in current_state.services:
            svc = current_state.services[action.service_name]
            count = action.count or 1

            if action.action_type == "scale_up":
                svc.replicas = min(10, svc.replicas + count)
                # CPU drops as load gets distributed across more replicas.
                svc.cpu_utilization = max(10.0, svc.cpu_utilization - (15.0 * count))
                action_cost = 0.5 * count  # Cost for scaling up (operational overhead)
            elif action.action_type == "scale_down":
                svc.replicas = max(1, svc.replicas - count)
                # CPU rises when load concentrates on fewer replicas.
                svc.cpu_utilization = min(100.0, svc.cpu_utilization + (25.0 * count))
            elif action.action_type == "restart":
                svc.status = "healthy"
                svc.memory_utilization = 20.0  # Restart clears leak accumulation.
                current_state.restart_count += 1
                action_cost = 0.3  # Cost for restarting (downtime/overhead)
        elif action.action_type != "do_nothing":
            invalid_action = True
            current_state.invalid_action_count += 1
        
        current_state.action_costs += action_cost
        current_state.total_cost = current_state.action_costs

        # 2. Simulate environment dynamics: traffic spikes and memory leaks.
        # Calculate current budget utilization
        current_budget = sum(service.replicas for service in current_state.services.values())
        budget_factor = current_budget / max(1, current_state.max_steps)  # Budget exhaustion stress
        
        for svc_name, svc in current_state.services.items():
            if svc.status != "crashed":
                # Simulated traffic fluctuations (-5% to +15%, amplified under high budget use)
                traffic_variance = self._rng.uniform(-5, 15) * (1.0 + budget_factor * 0.2)
                svc.cpu_utilization = min(100.0, max(0.0, svc.cpu_utilization + traffic_variance))
                
                # Simulated memory leak (+2% to +5% per step, amplified if CPU is high)
                cpu_stress_factor = 1.0 + (svc.cpu_utilization / 100.0) * 0.3  # High CPU → memory pressure
                memory_increase = self._rng.uniform(2, 5) * cpu_stress_factor
                svc.memory_utilization = min(100.0, svc.memory_utilization + memory_increase)
                
                # Memory impacts CPU: high memory creates I/O pressure
                if svc.memory_utilization > 80.0:
                    svc.cpu_utilization = min(100.0, svc.cpu_utilization + (svc.memory_utilization - 80.0) * 0.1)

                if 85.0 <= svc.memory_utilization <= 95.0:
                    svc.status = "degraded"
                elif svc.status == "degraded" and svc.memory_utilization < 85.0:
                    svc.status = "healthy"

                if svc.memory_utilization > 95.0 or svc.cpu_utilization >= 100.0:
                    if svc.status != "crashed":
                        current_state.crash_count += 1
                    svc.status = "crashed"
                    svc.cpu_utilization = 0.0
                
                # Track SLA: CPU should be 50-70% for optimal operation
                if not (50.0 <= svc.cpu_utilization <= 70.0):
                    current_state.sla_violations += 1
                
                # Track uptime
                if svc.status == "healthy":
                    current_state.service_uptime[svc_name] += 1
            else:
                current_state.sla_violations += 1  # Crashed = SLA violation
        
        # 3. Handle service dependencies: if a dependency crashes, dependent service degrades
        for svc_name, dependencies in current_state.service_dependencies.items():
            svc = current_state.services[svc_name]
            for dep_name in dependencies:
                if dep_name in current_state.services:
                    dep_svc = current_state.services[dep_name]
                    if dep_svc.status == "crashed" and svc.status == "healthy":
                        svc.status = "degraded"
                        svc.cpu_utilization = min(100.0, svc.cpu_utilization + 20.0)

        current_state.step_count += 1
        current_state.total_budget_used += sum(service.replicas for service in current_state.services.values())
        cpu_deviation = sum(abs(service.cpu_utilization - self._target_cpu) for service in current_state.services.values())
        current_state.cpu_deviation_sum += cpu_deviation
        denom = current_state.step_count * max(1, len(current_state.services))
        current_state.avg_cpu_deviation = round(current_state.cpu_deviation_sum / denom, 4)
        
        # Calculate uptime percentage
        total_possible_uptime = current_state.step_count * len(current_state.services)
        actual_uptime = sum(current_state.service_uptime.values())
        current_state.uptime_percent = round((actual_uptime / total_possible_uptime * 100) if total_possible_uptime > 0 else 100.0, 2)

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
        """Dense reward signal balancing health, utilization, cost, and SLA compliance."""
        if self._state is None:
            raise RuntimeError("Environment is not initialized.")
        state = self._state

        cpu_balance = 0.0
        memory_safety = 0.0
        reliability = 0.0
        budget_efficiency = 0.0
        sla_compliance = 0.0
        cost_efficiency = 0.0

        # Per-service metrics
        num_services = len(state.services)
        for svc in state.services.values():
            if svc.status == "crashed":
                reliability -= 1.0
            else:
                # CPU balance: reward for staying near target
                cpu_diff = abs(svc.cpu_utilization - self._target_cpu)
                cpu_balance += max(0.0, 1.0 - (cpu_diff / 40.0))
                
                # Memory safety: higher utilization = worse
                memory_safety += max(0.0, 1.0 - (svc.memory_utilization / 100.0))
                
                # Reliability: healthy > degraded
                reliability += 0.5 if svc.status == "healthy" else 0.1
                
                # Budget efficiency: fewer replicas is better
                budget_efficiency += max(0.0, 1.0 - (svc.replicas - 1) / 9.0)
                
                # SLA compliance: CPU in 50-70% range
                if 50.0 <= svc.cpu_utilization <= 70.0:
                    sla_compliance += 0.2
        
        # Normalize by number of services
        cpu_balance /= num_services
        memory_safety /= num_services
        reliability /= num_services
        budget_efficiency /= num_services
        sla_compliance /= num_services
        
        # Cost efficiency: penalize high action costs
        cost_efficiency = max(0.0, 1.0 - (state.action_costs / 50.0))
        
        # Uptime efficiency: reward high uptime percentage
        uptime_reward = state.uptime_percent / 100.0 * 0.5
        
        # Decision timeliness: reward for proactive actions
        timeliness = 0.0
        if state.restart_count > 0:
            crash_ratio = state.crash_count / state.restart_count
            timeliness = max(0.0, 1.0 - crash_ratio * 0.3)
        elif state.crash_count == 0:
            timeliness = 0.3
        
        # Action efficiency: penalize excessive actions
        max_expected_actions = state.max_steps / 5
        total_actions = state.restart_count
        action_efficiency = max(0.0, 1.0 - (total_actions / max(1, max_expected_actions)))
        
        # Invalid action penalty
        invalid_penalty = -0.5 if invalid_action else 0.0
        
        # Weighted total: state + decision quality
        total = (
            cpu_balance * 0.22 +
            memory_safety * 0.18 +
            reliability * 0.18 +
            budget_efficiency * 0.12 +
            sla_compliance * 0.12 +
            cost_efficiency * 0.04 +
            timeliness * 0.08 +
            action_efficiency * 0.04 +
            uptime_reward +
            invalid_penalty
        )
        
        return Reward(
            value=round(total, 4),
            cpu_balance=round(cpu_balance, 4),
            memory_safety=round(memory_safety, 4),
            reliability=round(reliability, 4),
            budget_efficiency=round(budget_efficiency, 4),
            invalid_action_penalty=invalid_penalty,
        )

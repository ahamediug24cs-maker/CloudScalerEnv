from typing import Dict, Literal, Optional

from pydantic import BaseModel, Field


class ServiceState(BaseModel):
    replicas: int = Field(ge=1, le=10, description="Current number of running instances.")
    cpu_utilization: float = Field(ge=0.0, le=100.0, description="Average CPU usage percentage.")
    memory_utilization: float = Field(ge=0.0, le=100.0, description="Average memory usage percentage.")
    status: Literal["healthy", "degraded", "crashed"] = "healthy"


class Observation(BaseModel):
    step_count: int = 0
    max_steps: int = 20
    services: Dict[str, ServiceState]
    total_budget_used: float = 0.0
    crash_count: int = 0
    restart_count: int = 0
    invalid_action_count: int = 0
    avg_cpu_deviation: float = 0.0


class EnvState(Observation):
    task_id: str = "unknown"
    seed: int = 0
    cumulative_reward: float = 0.0
    _cpu_deviation_sum: float = 0.0


class Action(BaseModel):
    action_type: Literal["scale_up", "scale_down", "restart", "do_nothing"]
    service_name: Optional[str] = Field(None, description="Target service (for example, auth-service).")
    count: Optional[int] = Field(1, ge=1, le=3, description="Number of replicas to add or remove.")


class Reward(BaseModel):
    value: float
    cpu_balance: float = 0.0
    memory_safety: float = 0.0
    reliability: float = 0.0
    budget_efficiency: float = 0.0
    invalid_action_penalty: float = 0.0


class TaskSpec(BaseModel):
    task_id: str
    name: str
    difficulty: Literal["easy", "medium", "hard"]
    objective: str
    max_steps: int = Field(ge=5, le=100)
    seed: int = 0

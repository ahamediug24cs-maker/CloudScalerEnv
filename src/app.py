from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from .env import CloudScalerEnv
from .models import Action, ServiceState
from .tasks import get_task_easy, get_task_hard, get_task_medium


class ResetRequest(BaseModel):
    task_id: Optional[str] = None
    seed: Optional[int] = None
    max_steps: Optional[int] = None
    services: Optional[Dict[str, ServiceState]] = None


app = FastAPI(title="CloudScalerEnv OpenEnv API")
env = CloudScalerEnv()


def _resolve_task(task_id: Optional[str]):
    selected = task_id or "easy-memory-leak"
    if selected == "easy-memory-leak":
        return get_task_easy()
    if selected == "medium-traffic-spike":
        return get_task_medium()
    if selected == "hard-cascading-failure":
        return get_task_hard()
    raise HTTPException(status_code=400, detail=f"Unknown task_id: {selected}")


def _merge_services(
    base_services: Dict[str, ServiceState],
    override_services: Optional[Dict[str, ServiceState]],
) -> Dict[str, ServiceState]:
    if not override_services:
        return base_services

    merged: Dict[str, ServiceState] = {
        name: service.model_copy(deep=True) for name, service in base_services.items()
    }
    for name, service in override_services.items():
        merged[name] = service.model_copy(deep=True)
    return merged


@app.get("/")
async def root() -> dict:
    return {
        "name": "CloudScalerEnv",
        "status": "ok",
        "message": "Service is running.",
        "endpoints": {
            "health": "/health",
            "reset": "/reset",
            "step": "/step",
            "state": "/state",
        },
    }


@app.get("/health")
async def health() -> dict:
    return {"healthy": True}


@app.post("/reset")
@app.post("/reset/")
async def reset(payload: Optional[ResetRequest] = None) -> dict:
    try:
        task_id = payload.task_id if payload else None
        seed_override = payload.seed if payload else None
        task, initial_services, _grader = _resolve_task(task_id)
        selected_services = _merge_services(initial_services, payload.services if payload else None)
        selected_max_steps = payload.max_steps if payload and payload.max_steps is not None else task.max_steps

        if selected_max_steps < 5 or selected_max_steps > 100:
            raise HTTPException(status_code=400, detail="max_steps must be in range [5, 100]")

        obs = env.reset(
            initial_services=selected_services,
            max_steps=selected_max_steps,
            task_id=task.task_id,
            seed=(seed_override if seed_override is not None else task.seed),
        )
        return obs.model_dump()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(exc)}")


@app.post("/step")
@app.post("/step/")
async def step(action: Action) -> dict:
    try:
        obs, reward, done, info = env.step(action)
        return {
            "observation": obs.model_dump(),
            "reward": reward.model_dump(),
            "done": done,
            "info": info,
        }
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Step failed: {str(exc)}")


@app.get("/state")
@app.get("/state/")
async def state() -> dict:
    try:
        return env.state().model_dump()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"State failed: {str(exc)}")


def run_server() -> None:
    uvicorn.run("src.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    run_server()
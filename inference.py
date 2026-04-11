import json
import os
import sys
import textwrap
from typing import List, Optional, Tuple

from openai import OpenAI

from src.env import CloudScalerEnv
from src.models import Action, Observation
from src.policy import choose_action
from src.tasks import get_task_easy, get_task_hard, get_task_medium

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

BENCHMARK_NAME = os.getenv("BENCHMARK_NAME", "cloudscalerenv")
TASK_NAME = os.getenv("TASK_NAME", "easy-memory-leak")

TEMPERATURE = 0.0
MAX_TOKENS = 220

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an SRE agent for microservices.
    Choose one action to improve reliability and cost efficiency.
    Return only one compact JSON object with keys:
    action_type, service_name, count.
    action_type must be one of: scale_up, scale_down, restart, do_nothing.
    """
).strip()


def _get_task(task_name: str):
    if task_name == "easy-memory-leak":
        return get_task_easy()
    if task_name == "medium-traffic-spike":
        return get_task_medium()
    if task_name == "hard-cascading-failure":
        return get_task_hard()
    raise ValueError(f"Unsupported TASK_NAME: {task_name}")


def _obs_to_prompt(obs: Observation) -> str:
    services = []
    for name, svc in obs.services.items():
        services.append(
            {
                "name": name,
                "replicas": svc.replicas,
                "cpu": round(svc.cpu_utilization, 2),
                "memory": round(svc.memory_utilization, 2),
                "status": svc.status,
            }
        )

    payload = {
        "step": obs.step_count,
        "max_steps": obs.max_steps,
        "services": services,
        "total_budget_used": round(obs.total_budget_used, 2),
        "crash_count": obs.crash_count,
        "restart_count": obs.restart_count,
        "invalid_action_count": obs.invalid_action_count,
    }
    return json.dumps(payload, separators=(",", ":"))


def _sanitize_action(candidate: Optional[Action], obs: Observation, task_name: str) -> Action:
    if candidate is None:
        return choose_action(obs, task_name)

    if candidate.action_type == "do_nothing":
        return Action(action_type="do_nothing")

    service_name = candidate.service_name
    if not service_name or service_name not in obs.services:
        return choose_action(obs, task_name)

    count = candidate.count or 1
    if count < 1:
        count = 1
    if count > 3:
        count = 3

    return Action(action_type=candidate.action_type, service_name=service_name, count=count)


def _model_action(client: OpenAI, obs: Observation, task_name: str) -> Action:
    prompt = _obs_to_prompt(obs)

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        text = (completion.choices[0].message.content or "").strip()
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return choose_action(obs, task_name)

        parsed = json.loads(text[start : end + 1])
        action = Action.model_validate(parsed)
        return _sanitize_action(action, obs, task_name)
    except Exception:
        return choose_action(obs, task_name)


def _action_str(action: Action) -> str:
    if action.action_type == "do_nothing":
        return "do_nothing"
    svc = action.service_name if action.service_name else ""
    cnt = action.count if action.count is not None else 1
    return f"{action.action_type}({svc},{cnt})"


def _log_start(task_name: str) -> None:
    print(f"[START] task={task_name} env={BENCHMARK_NAME} model={MODEL_NAME}", flush=True)


def _log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    err = "null" if error is None else str(error).replace("\n", " ").replace("\r", " ")
    done_str = "true" if done else "false"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_str} error={err}",
        flush=True,
    )


def _log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    success_str = "true" if success else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
    print(
        f"[END] success={success_str} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def run_episode() -> Tuple[bool, int, List[float]]:
    rewards: List[float] = []
    steps = 0
    success = False
    episode_score = 0.0
    env: Optional[CloudScalerEnv] = None

    _log_start(TASK_NAME)

    try:
        if HF_TOKEN is None:
            raise ValueError("HF_TOKEN environment variable is required")

        client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN, timeout=5.0, max_retries=0)
        env = CloudScalerEnv()

        task, initial_services, _grader = _get_task(TASK_NAME)
        obs = env.reset_for_task(task, initial_services)

        done = False
        while not done:
            action = _model_action(client, obs, TASK_NAME)
            obs, reward, done, info = env.step(action)

            steps += 1
            reward_value = reward.value if reward is not None else 0.0
            rewards.append(reward_value)

            raw_error = info.get("last_action_error") if isinstance(info, dict) else None
            _log_step(
                step=steps,
                action=_action_str(action),
                reward=reward_value,
                done=done,
                error=raw_error,
            )

        success = True
        if rewards:
            episode_score = max(0.0, min(1.0, sum(rewards) / len(rewards)))
        return success, steps, rewards
    except Exception as exc:
        print(f"inference_error: {exc}", file=sys.stderr)
        return success, steps, rewards
    finally:
        if env is not None:
            close_fn = getattr(env, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass

        if rewards:
            episode_score = max(0.0, min(1.0, sum(rewards) / len(rewards)))
        _log_end(success=success, steps=steps, score=episode_score, rewards=rewards)


def main() -> None:
    run_episode()


if __name__ == "__main__":
    main()

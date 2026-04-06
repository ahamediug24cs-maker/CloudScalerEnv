import argparse
import json
import os
from typing import Dict

from openai import OpenAI

from src.env import CloudScalerEnv
from src.models import Action, Observation
from src.tasks import get_task_easy, get_task_hard, get_task_medium


def heuristic_agent(state: Observation) -> Action:
    """Heuristic policy: restart high memory, scale high CPU, scale down low CPU."""
    for name, svc in state.services.items():
        if svc.memory_utilization > 80.0:
            return Action(action_type="restart", service_name=name)
        if svc.cpu_utilization > 80.0:
            return Action(action_type="scale_up", service_name=name, count=1)
        if svc.cpu_utilization < 30.0 and svc.replicas > 1:
            return Action(action_type="scale_down", service_name=name, count=1)
    return Action(action_type="do_nothing")


def _state_to_prompt(state: Observation) -> str:
    services = []
    for name, svc in state.services.items():
        services.append(
            {
                "name": name,
                "replicas": svc.replicas,
                "cpu_utilization": round(svc.cpu_utilization, 2),
                "memory_utilization": round(svc.memory_utilization, 2),
                "status": svc.status,
            }
        )
    payload = {
        "step_count": state.step_count,
        "max_steps": state.max_steps,
        "services": services,
        "total_budget_used": round(state.total_budget_used, 2),
        "crash_count": state.crash_count,
        "restart_count": state.restart_count,
        "invalid_action_count": state.invalid_action_count,
    }
    return json.dumps(payload, indent=2)


def llm_agent(client: OpenAI, model: str, state: Observation) -> Action:
    prompt = (
        "You are an SRE agent managing microservice reliability and cost. "
        "Keep CPU around 50-70%, avoid crashes, and avoid wasteful scaling. "
        "Return ONLY JSON with keys: action_type, service_name, count. "
        "Valid action_type values: scale_up, scale_down, restart, do_nothing.\n\n"
        "Current state:\n"
        f"{_state_to_prompt(state)}"
    )

    response = client.responses.create(
        model=model,
        temperature=0,
        input=[
            {
                "role": "system",
                "content": "You output strict JSON only and never include markdown.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    text = response.output_text.strip()
    try:
        parsed = json.loads(text)
        return Action.model_validate(parsed)
    except Exception:
        return Action(action_type="do_nothing")


def run_baseline(mode: str = "heuristic", model: str = "gpt-4.1-mini") -> Dict[str, float]:
    env = CloudScalerEnv()
    tasks = [("Easy", get_task_easy), ("Medium", get_task_medium), ("Hard", get_task_hard)]
    scores: Dict[str, float] = {}

    client = None
    if mode == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required when mode=openai.")
        client = OpenAI(api_key=api_key)

    for task_name, task_fn in tasks:
        task, initial_state, grader = task_fn()
        state = env.reset_for_task(task, initial_state)
        done = False

        while not done:
            if mode == "openai":
                action = llm_agent(client=client, model=model, state=state)
            else:
                action = heuristic_agent(state)
            state, reward, done, _ = env.step(action)
            _ = reward.value

        final_state = env.raw_state()
        score = grader.grade(final_state)
        scores[task_name] = score
        print(f"Task: {task_name} | Grader Score (0.0-1.0): {score}")

    overall = round(sum(scores.values()) / len(scores), 3)
    print(f"Overall Mean Score: {overall}")
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run baseline agents against CloudScalerEnv tasks.")
    parser.add_argument(
        "--mode",
        choices=["heuristic", "openai"],
        default="heuristic",
        help="Use heuristic policy or OpenAI API policy.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        help="OpenAI model name when --mode openai.",
    )
    args = parser.parse_args()
    run_baseline(mode=args.mode, model=args.model)

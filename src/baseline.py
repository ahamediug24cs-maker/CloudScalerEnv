import argparse
import json
import os
from typing import Dict, List, Optional

from openai import OpenAI

from src.env import CloudScalerEnv
from src.models import Action, Observation
from src.policy import choose_action
from src.tasks import get_task_easy, get_task_hard, get_task_medium


def heuristic_agent(state: Observation, task_name: str) -> Action:
    """Task-aware heuristic shared with the inference runner."""
    return choose_action(state, task_name)


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


def _parse_seed_sweep(seed_sweep: str) -> List[int]:
    values: List[int] = []
    for raw in seed_sweep.split(","):
        token = raw.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError("seed-sweep must contain at least one integer seed")
    return values


def run_baseline(
    mode: str = "heuristic",
    model: str = "gpt-4.1-mini",
    seed: Optional[int] = None,
) -> Dict[str, float]:
    env = CloudScalerEnv()
    tasks = [
        ("Easy", get_task_easy),
        ("Medium", get_task_medium),
        ("Hard", get_task_hard),
    ]
    scores: Dict[str, float] = {}

    client = None
    if mode == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required when mode=openai.")
        client = OpenAI(api_key=api_key)

    for task_name, task_fn in tasks:
        task, initial_state, grader = task_fn()
        if seed is None:
            state = env.reset_for_task(task, initial_state)
        else:
            state = env.reset(
                initial_services=initial_state,
                max_steps=task.max_steps,
                task_id=task.task_id,
                seed=seed,
            )
        done = False

        while not done:
            if mode == "openai":
                action = llm_agent(client=client, model=model, state=state)
            else:
                action = heuristic_agent(state, task.task_id)
            state, reward, done, _ = env.step(action)
            _ = reward.value

        final_state = env.raw_state()
        score = grader.grade(final_state)
        scores[task_name] = score
        seed_label = seed if seed is not None else task.seed
        print(f"Task: {task_name} | Seed: {seed_label} | Grader Score (0.0-1.0): {score}")

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
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override task seed with a single integer for all tasks.",
    )
    parser.add_argument(
        "--seed-sweep",
        type=str,
        default=None,
        help="Comma-separated seeds to run sequentially, for example: 11,22,33",
    )
    args = parser.parse_args()

    if args.seed_sweep:
        seeds = _parse_seed_sweep(args.seed_sweep)
        for i, seed in enumerate(seeds, start=1):
            print(f"\n=== Seed Sweep Run {i}/{len(seeds)} | seed={seed} ===")
            run_baseline(mode=args.mode, model=args.model, seed=seed)
    else:
        run_baseline(mode=args.mode, model=args.model, seed=args.seed)

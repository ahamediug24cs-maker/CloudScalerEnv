"""
CloudScalerEnv Inference Script
=============================

MANDATORY REQUIREMENTS:
- Environment variables:
    API_BASE_URL   The API endpoint for the LLM (default: https://router.huggingface.co/v1)
    MODEL_NAME     The model identifier (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN       Your Hugging Face / API key
    LOCAL_IMAGE_NAME The local Docker image name (optional, for docker-based inference)

- Uses OpenAI Client for all LLM calls
- Emits three line types to stdout:
    [START] task=<task_name> env=cloudscalerenv model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import json
import os
import random
import textwrap
from typing import List, Optional

from openai import OpenAI

from src.env import CloudScalerEnv
from src.models import Action, Observation
from src.policy import choose_action
from src.tasks import get_task_easy, get_task_hard, get_task_medium

# Environment configuration
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

# Hyperparameters
MAX_STEPS = 30
TEMPERATURE = 0.2
MAX_TOKENS = 256

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an SRE agent managing microservice reliability and cost optimization.
    Given the current state of services (CPU utilization, memory usage, replica counts),
    decide the best action to maintain stability while controlling costs.
    
    Available actions:
    - scale_up: increase replicas to distribute load
    - scale_down: decrease replicas to save costs
    - restart: restart service to clear memory leaks
    - do_nothing: maintain current state
    
    Respond with ONLY a JSON object on a single line, no markdown:
    {"action_type": "scale_up"|"scale_down"|"restart"|"do_nothing", "service_name": "service-name", "count": 1}
    
    For do_nothing, service_name and count are optional.
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    """Emit [START] line."""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    """Emit [STEP] line."""
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """Emit [END] line."""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def state_to_prompt(obs: Observation) -> str:
    """Convert observation to natural language prompt."""
    services_desc = []
    for name, svc in obs.services.items():
        services_desc.append(
            f"  {name}: {svc.replicas} replicas, CPU {svc.cpu_utilization:.1f}%, Memory {svc.memory_utilization:.1f}%, Status: {svc.status}"
        )
    
    return textwrap.dedent(
        f"""
        Step {obs.step_count}/{obs.max_steps}
        Services:
        {chr(10).join(services_desc)}
        Budget used: {obs.total_budget_used:.0f} replica-steps
        Crashes so far: {obs.crash_count}
        
        What action should we take?
        """
    ).strip()


def _sanitize_action(candidate: Optional[Action], obs: Observation, task_name: str) -> Action:
    """Reject unsafe/invalid choices and fall back to robust heuristic."""
    if task_name == "medium-traffic-spike":
        upper_guard = 66.0
    elif task_name == "hard-cascading-failure":
        upper_guard = 68.0
    else:
        upper_guard = 70.0

    if candidate is None:
        return choose_action(obs, task_name)

    if candidate.action_type == "do_nothing":
        return Action(action_type="do_nothing")

    svc_name = candidate.service_name
    if not svc_name or svc_name not in obs.services:
        return choose_action(obs, task_name)

    svc = obs.services[svc_name]
    count = candidate.count or 1
    if count < 1:
        count = 1
    if count > 3:
        count = 3

    if candidate.action_type == "scale_up" and svc.replicas >= 10:
        return Action(action_type="do_nothing")
    if candidate.action_type == "scale_down":
        if svc.replicas <= 1 or svc.cpu_utilization > upper_guard:
            return choose_action(obs, task_name)
    if candidate.action_type == "restart":
        healthy_and_safe = svc.status == "healthy" and svc.memory_utilization < 78.0 and 50.0 <= svc.cpu_utilization <= 70.0
        if healthy_and_safe:
            return choose_action(obs, task_name)

    return Action(action_type=candidate.action_type, service_name=svc_name, count=count)


def get_model_action(client: Optional[OpenAI], obs: Observation, task_name: str) -> Optional[Action]:
    """Query LLM to get next action. Falls back to heuristic if no client."""
    if client is None:
        return choose_action(obs, task_name)
    
    prompt = state_to_prompt(obs)
    
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        
        # Try to extract JSON object from plain text.
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            parsed = json.loads(text[start : end + 1])
            action = Action.model_validate(parsed)
            return _sanitize_action(action, obs, task_name)
        return choose_action(obs, task_name)
    except Exception as exc:
        return choose_action(obs, task_name)


def run_task(task_name: str, task_fn) -> tuple[bool, int, float, List[float]]:
    """Run a single task and return (success, steps, score, rewards)."""
    # Create client only if API key is available
    client = None
    if HF_TOKEN:
        try:
            client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
        except Exception:
            client = None
    
    env = CloudScalerEnv()
    task, initial_services, grader = task_fn()
    
    # Randomize seed each run for realistic variation
    task.seed = random.randint(1000, 9999)
    
    log_start(task=task_name, env="cloudscalerenv", model=MODEL_NAME)
    
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    
    try:
        obs = env.reset_for_task(task, initial_services)
        
        for step in range(1, MAX_STEPS + 1):
            if obs.step_count >= obs.max_steps:
                break
            
            # Get action from LLM or heuristic
            action = get_model_action(client, obs, task_name)
            if action is None:
                action_str = "do_nothing"
                action = Action(action_type="do_nothing")
            else:
                action_str = f"{action.action_type}(service={action.service_name}, count={action.count})"
            
            # Step environment
            obs, reward, done, info = env.step(action)
            
            reward_value = reward.value if reward else 0.0
            rewards.append(reward_value)
            steps_taken = step
            
            log_step(step=step, action=action_str, reward=reward_value, done=done, error=None)
            
            if done:
                break
        
        # Grade final trajectory
        final_state = env.raw_state()
        score = grader.grade(final_state)
        success = score >= 0.3  # threshold for "success"
        
    except Exception as exc:
        score = 0.0
        success = False
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    
    return success, steps_taken, score, rewards


def main():
    """Run inference across all three tasks."""
    tasks = [
        ("easy-memory-leak", get_task_easy),
        ("medium-traffic-spike", get_task_medium),
        ("hard-cascading-failure", get_task_hard),
    ]
    
    for task_name, task_fn in tasks:
        run_task(task_name, task_fn)


if __name__ == "__main__":
    main()

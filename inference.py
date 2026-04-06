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
import textwrap
from typing import List, Optional

from openai import OpenAI

from src.env import CloudScalerEnv
from src.models import Action, Observation
from src.tasks import get_task_easy, get_task_hard, get_task_medium

# Environment configuration
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

# Hyperparameters
MAX_STEPS = 30
TEMPERATURE = 0.7
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


def get_model_action(client: Optional[OpenAI], obs: Observation) -> Optional[Action]:
    """Query LLM to get next action. Falls back to heuristic if no client."""
    if client is None:
        # Fallback heuristic
        for name, svc in obs.services.items():
            if svc.memory_utilization > 80.0:
                return Action(action_type="restart", service_name=name)
            if svc.cpu_utilization > 80.0:
                return Action(action_type="scale_up", service_name=name, count=1)
            if svc.cpu_utilization < 30.0 and svc.replicas > 1:
                return Action(action_type="scale_down", service_name=name, count=1)
        return Action(action_type="do_nothing")
    
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
        
        # Try to extract JSON
        if text.startswith("{"):
            parsed = json.loads(text)
            action = Action.model_validate(parsed)
            return action
        else:
            # Fallback to do_nothing
            return Action(action_type="do_nothing")
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return Action(action_type="do_nothing")


def run_task(task_name: str, task_fn) -> tuple[bool, int, float, List[float]]:
    """Run a single task and return (success, steps, score, rewards)."""
    # Create client only if API key is available
    client = None
    if HF_TOKEN:
        try:
            client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
        except Exception as e:
            print(f"[DEBUG] Failed to create OpenAI client: {e}, using heuristic fallback", flush=True)
    else:
        print(f"[DEBUG] No HF_TOKEN provided, using heuristic fallback", flush=True)
    
    env = CloudScalerEnv()
    task, initial_services, grader = task_fn()
    
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
            action = get_model_action(client, obs)
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
        print(f"[DEBUG] Task execution error: {exc}", flush=True)
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
    
    all_scores = []
    
    for task_name, task_fn in tasks:
        print(f"\n[INFO] Starting task: {task_name}", flush=True)
        success, steps, score, rewards = run_task(task_name, task_fn)
        all_scores.append(score)
        print(f"[INFO] Task complete: score={score:.3f}, success={success}, steps={steps}\n", flush=True)
    
    mean_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
    print(f"[SUMMARY] Mean score across all tasks: {mean_score:.3f}", flush=True)


if __name__ == "__main__":
    main()

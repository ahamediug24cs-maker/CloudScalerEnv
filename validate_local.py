#!/usr/bin/env python3
"""
Local OpenEnv validator: checks compliance without relying on external CLI.
Verifies metadata, imports, interface contract, and basic functionality.
"""

import sys
from pathlib import Path

import yaml

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.env import CloudScalerEnv
from src.models import Action, Observation, Reward, ServiceState
from src.policy import choose_action
from src.tasks import get_task_easy, get_task_medium, get_task_hard


def validate_metadata() -> bool:
    """Check openenv.yaml structure and content."""
    print("Validating metadata (openenv.yaml)...")
    metadata_path = Path(__file__).parent / "openenv.yaml"
    
    if not metadata_path.exists():
        print(f"  ✗ openenv.yaml not found at {metadata_path}")
        return False
    
    with open(metadata_path) as f:
        metadata = yaml.safe_load(f)
    
    required_keys = {"name", "version", "description", "environment", "tasks"}
    if not required_keys.issubset(metadata.keys()):
        print(f"  ✗ Missing required keys: {required_keys - set(metadata.keys())}")
        return False
    
    env_spec = metadata.get("environment", {})
    required_env_keys = {"entrypoint", "state_model", "action_model", "reward_model"}
    if not required_env_keys.issubset(env_spec.keys()):
        print(f"  ✗ Missing env keys: {required_env_keys - set(env_spec.keys())}")
        return False
    
    if env_spec["entrypoint"] != "src.env:CloudScalerEnv":
        print(f"  ✗ Wrong entrypoint: {env_spec['entrypoint']}")
        return False
    
    if env_spec["state_model"] != "src.models:Observation":
        print(f"  ✗ Wrong state_model: {env_spec['state_model']}")
        return False
    
    if env_spec["action_model"] != "src.models:Action":
        print(f"  ✗ Wrong action_model: {env_spec['action_model']}")
        return False
    
    if env_spec["reward_model"] != "src.models:Reward":
        print(f"  ✗ Wrong reward_model: {env_spec['reward_model']}")
        return False
    
    tasks = metadata.get("tasks", [])
    if len(tasks) < 3:
        print(f"  ✗ Found {len(tasks)} tasks, expected 3+")
        return False
    
    print("  ✓ Metadata valid")
    return True


def validate_models() -> bool:
    """Check that models are valid Pydantic models."""
    print("Validating models...")
    
    try:
        svc = ServiceState(replicas=2, cpu_utilization=50.0, memory_utilization=70.0, status="healthy")
        print(f"  ✓ ServiceState instantiated: {svc.model_fields_set}")
        
        obs = Observation(
            step_count=0, max_steps=20, services={"test": svc},
            total_budget_used=2.0, crash_count=0, restart_count=0,
            invalid_action_count=0, avg_cpu_deviation=0.0
        )
        print(f"  ✓ Observation instantiated")
        
        action = Action(action_type="scale_up", service_name="test", count=1)
        print(f"  ✓ Action instantiated: {action}")
        
        reward = Reward(value=0.5, cpu_balance=0.2, memory_safety=0.15, reliability=0.1, budget_efficiency=0.05)
        print(f"  ✓ Reward instantiated: value={reward.value}")
        
    except Exception as e:
        print(f"  ✗ Model validation failed: {e}")
        return False
    
    return True


def validate_environment_interface() -> bool:
    """Check that environment implements reset, step, state methods."""
    print("Validating environment interface...")
    
    try:
        env = CloudScalerEnv()
        
        if not hasattr(env, "reset"):
            print("  ✗ Missing reset() method")
            return False
        if not hasattr(env, "step"):
            print("  ✗ Missing step() method")
            return False
        if not hasattr(env, "state"):
            print("  ✗ Missing state() method")
            return False
        
        print("  ✓ All required methods present")
        
    except Exception as e:
        print(f"  ✗ Environment interface check failed: {e}")
        return False
    
    return True


def validate_episode_contract() -> bool:
    """Test reset -> step -> state lifecycle."""
    print("Validating episode contract...")
    
    try:
        env = CloudScalerEnv()
        initial_state = {
            "test-svc": ServiceState(replicas=2, cpu_utilization=50.0, memory_utilization=70.0)
        }
        
        # Reset
        obs = env.reset(initial_state, max_steps=5)
        if not isinstance(obs, Observation):
            print(f"  ✗ reset() returned {type(obs)}, expected Observation")
            return False
        print(f"  ✓ reset() returned Observation")
        
        # Step once
        action = Action(action_type="do_nothing")
        obs_next, reward, done, info = env.step(action)
        
        if not isinstance(obs_next, Observation):
            print(f"  ✗ step() returned obs type {type(obs_next)}, expected Observation")
            return False
        if not isinstance(reward, Reward):
            print(f"  ✗ step() returned reward type {type(reward)}, expected Reward")
            return False
        if not isinstance(done, bool):
            print(f"  ✗ step() returned done type {type(done)}, expected bool")
            return False
        if not isinstance(info, dict):
            print(f"  ✗ step() returned info type {type(info)}, expected dict")
            return False
        
        print(f"  ✓ step() contract valid: obs, reward, done={done}, info")
        
        # State
        state_check = env.state()
        if not isinstance(state_check, Observation):
            print(f"  ✗ state() returned {type(state_check)}, expected Observation")
            return False
        print(f"  ✓ state() returned Observation")
        
    except Exception as e:
        print(f"  ✗ Episode contract test failed: {e}")
        return False
    
    return True


def validate_tasks() -> bool:
    """Check that all 3 tasks load and graders work."""
    print("Validating tasks and graders...")
    
    try:
        tasks_list = [
            ("easy", get_task_easy),
            ("medium", get_task_medium),
            ("hard", get_task_hard),
        ]
        
        for name, task_fn in tasks_list:
            task, initial, grader = task_fn()
            
            if not hasattr(grader, "grade"):
                print(f"  ✗ {name} grader missing grade() method")
                return False
            
            env = CloudScalerEnv()
            obs = env.reset_for_task(task, initial)
            
            done = False
            steps = 0
            while not done and steps < 100:
                action = choose_action(obs, task.task_id)
                obs, reward, done, info = env.step(action)
                steps += 1
            
            final_state = env.raw_state()
            score = grader.grade(final_state)
            
            if not 0.0 <= score <= 1.0:
                print(f"  ✗ {name} grader score {score} out of range [0.0, 1.0]")
                return False
            
            print(f"  ✓ {name}: task loaded, grader score {score}")
        
    except Exception as e:
        print(f"  ✗ Task validation failed: {e}")
        return False
    
    return True


def main():
    print("=" * 60)
    print("CloudScalerEnv Local OpenEnv Validator")
    print("=" * 60)
    
    checks = [
        ("Metadata", validate_metadata),
        ("Models", validate_models),
        ("Environment Interface", validate_environment_interface),
        ("Episode Contract", validate_episode_contract),
        ("Tasks & Graders", validate_tasks),
    ]
    
    passed = 0
    for check_name, check_fn in checks:
        try:
            if check_fn():
                passed += 1
        except Exception as e:
            print(f"✗ {check_name} check failed with exception: {e}")
            import traceback
            traceback.print_exc()
    
    print("=" * 60)
    print(f"Result: {passed}/{len(checks)} checks passed")
    print("=" * 60)
    
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())

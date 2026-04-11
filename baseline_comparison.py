#!/usr/bin/env python3
"""
Comparison baseline: Show how heuristic policy performs vs random/do-nothing agents.
Useful for demonstrating competitive advantage.
"""

import random
from src.env import CloudScalerEnv
from src.models import Action
from src.policy import choose_action
from src.tasks import get_task_easy, get_task_medium, get_task_hard

TASKS = {
    "easy": get_task_easy,
    "medium": get_task_medium,
    "hard": get_task_hard,
}


class Agent:
    def __init__(self, name: str):
        self.name = name

    def select_action(self, obs, task_name: str) -> Action:
        raise NotImplementedError


class HeuristicAgent(Agent):
    """Smart baseline using policy rules."""
    def select_action(self, obs, task_name: str) -> Action:
        return choose_action(obs, task_name)


class RandomAgent(Agent):
    """Baseline: random actions with 30% probability."""
    def select_action(self, obs, task_name: str) -> Action:
        if random.random() > 0.3:
            return Action(action_type="do_nothing")
        
        services = list(obs.services.keys())
        service = random.choice(services)
        action_type = random.choice(["scale_up", "scale_down", "restart"])
        return Action(action_type=action_type, service_name=service, count=1)


class DoNothingAgent(Agent):
    """Baseline: always do nothing."""
    def select_action(self, obs, task_name: str) -> Action:
        return Action(action_type="do_nothing")


def run_episode(agent: Agent, task_name: str, seed_override: int | None = None) -> float:
    """Run a single episode and return final score."""
    task_fn = TASKS[task_name]
    task, initial, grader = task_fn()
    env = CloudScalerEnv()
    seed = task.seed if seed_override is None else seed_override
    # Ensure random baseline is reproducible for a given task+seed.
    random.seed(f"{task_name}:{seed}:{agent.name}")
    obs = env.reset(
        initial_services=initial,
        max_steps=task.max_steps,
        task_id=task.task_id,
        seed=seed,
    )
    
    done = False
    while not done:
        action = agent.select_action(obs, task_name)
        obs, reward, done, _ = env.step(action)
    
    final_state = env.raw_state()
    score = grader.grade(final_state)
    return score


def main():
    print("=" * 70)
    print("BASELINE COMPARISON: Heuristic vs Random vs Do-Nothing")
    print("=" * 70)
    
    agents = [
        HeuristicAgent("Heuristic Policy"),
        RandomAgent("Random Agent"),
        DoNothingAgent("Do-Nothing"),
    ]
    
    tasks = ["easy", "medium", "hard"]
    num_runs = 3
    
    results = {agent.name: {task: [] for task in tasks} for agent in agents}
    
    for task_name in tasks:
        print(f"\nTask: {task_name.upper()}")
        print("-" * 70)
        base_seed = TASKS[task_name]()[0].seed
        
        for agent in agents:
            scores = []
            for run in range(num_runs):
                seed = base_seed + (run * 17)
                score = run_episode(agent, task_name, seed_override=seed)
                scores.append(score)
            
            results[agent.name][task_name] = scores
            avg_score = sum(scores) / len(scores)
            print(f"{agent.name:25} | Avg: {avg_score:.4f} | Scores: {', '.join(f'{s:.4f}' for s in scores)}")
    
    print("\n" + "=" * 70)
    print("SUMMARY: Average Scores by Agent")
    print("=" * 70)
    for agent in agents:
        all_scores = []
        for task in tasks:
            all_scores.extend(results[agent.name][task])
        avg = sum(all_scores) / len(all_scores)
        print(f"{agent.name:25} | Overall Average: {avg:.4f}")
    
    print("\n" + "=" * 70)
    print("Performance Advantage (Heuristic - Random):")
    heuristic_avg = sum(sum(results["Heuristic Policy"][t]) for t in tasks) / (len(tasks) * num_runs)
    random_avg = sum(sum(results["Random Agent"][t]) for t in tasks) / (len(tasks) * num_runs)
    advantage = ((heuristic_avg - random_avg) / random_avg) * 100
    print(f"Heuristic is {advantage:.1f}% better than Random")
    print("=" * 70)


if __name__ == "__main__":
    main()

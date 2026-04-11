import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.client import CloudScalerClient
from src.models import Observation
from src.policy import choose_action


def main() -> None:
    server = subprocess.Popen(
        ["python", "-m", "src.app"],
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(3.0)
    client = CloudScalerClient()
    task_id = "hard-cascading-failure"

    try:
        obs_dict = client.reset(task_id=task_id, seed=33)
        done = False
        steps = 0

        while not done and steps < 40:
            obs = Observation.model_validate(obs_dict)
            action = choose_action(obs, task_id)
            result = client.step(action.model_dump())

            steps += 1
            obs_dict = result["observation"]
            done = result["done"]
            reward = result["reward"]["value"]
            print(f"Step {steps}: action={action.action_type}, reward={reward:.2f}, done={done}")

        state = client.state()
        print("Summary:", {
            "steps": state["step_count"],
            "crashes": state["crash_count"],
            "sla_violations": state["sla_violations"],
            "uptime_percent": state["uptime_percent"],
        })
    finally:
        client.close()
        server.terminate()
        try:
            server.wait(timeout=5)
        except Exception:
            server.kill()


if __name__ == "__main__":
    main()

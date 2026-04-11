import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.client import CloudScalerClient


def main() -> None:
    server = subprocess.Popen(
        ["python", "-m", "src.app"],
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(3.0)
    client = CloudScalerClient()
    try:
        print("Health:", client.health())

        obs = client.reset(task_id="easy-memory-leak", seed=11)
        print("Reset step_count:", obs["step_count"])

        actions = [
            {"action_type": "do_nothing"},
            {"action_type": "restart", "service_name": "web-frontend", "count": 1},
            {"action_type": "do_nothing"},
        ]

        for i, action in enumerate(actions, start=1):
            result = client.step(action)
            reward = result["reward"]["value"]
            done = result["done"]
            print(f"Step {i}: reward={reward:.2f}, done={done}")
            if done:
                break

        state = client.state()
        print("Final crash_count:", state["crash_count"])
    finally:
        client.close()
        server.terminate()
        try:
            server.wait(timeout=5)
        except Exception:
            server.kill()


if __name__ == "__main__":
    main()

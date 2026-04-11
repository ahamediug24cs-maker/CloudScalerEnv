import asyncio
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.client import AsyncCloudScalerClient


async def run_episode() -> None:
    client = AsyncCloudScalerClient()
    try:
        print("Health:", await client.health())

        obs = await client.reset(task_id="medium-traffic-spike", seed=22)
        print("Reset step_count:", obs["step_count"])

        result = await client.step({"action_type": "do_nothing"})
        print("Step reward:", round(result["reward"]["value"], 2))
        print("Done:", result["done"])

        state = await client.state()
        print("State step_count:", state["step_count"])
    finally:
        await client.close()


def main() -> None:
    server = subprocess.Popen(
        ["python", "-m", "src.app"],
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(3.0)
    try:
        asyncio.run(run_episode())
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except Exception:
            server.kill()


if __name__ == "__main__":
    main()

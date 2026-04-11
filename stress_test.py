import argparse
import asyncio
import json
import statistics
import subprocess
import time
from pathlib import Path

import httpx


ROOT = Path(__file__).resolve().parent
BASE_URL = "http://127.0.0.1:7860"


async def _worker(client: httpx.AsyncClient, task_id: str, requests_per_worker: int) -> dict:
    latencies = []
    ok = 0
    errors = 0

    for _ in range(requests_per_worker):
        start = time.perf_counter()
        try:
            r1 = await client.post("/reset", json={"task_id": task_id})
            r2 = await client.post("/step", json={"action_type": "do_nothing"})
            r3 = await client.get("/state")
            elapsed = (time.perf_counter() - start) * 1000.0
            if r1.status_code == 200 and r2.status_code == 200 and r3.status_code == 200:
                ok += 1
                latencies.append(elapsed)
            else:
                errors += 1
        except Exception:
            errors += 1

    return {"ok": ok, "errors": errors, "latencies_ms": latencies}


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    idx = min(len(values) - 1, max(0, int(round((p / 100.0) * (len(values) - 1)))))
    return sorted(values)[idx]


async def run_stress(task_id: str, concurrency: int, requests_per_worker: int) -> dict:
    timeout = httpx.Timeout(connect=10.0, read=20.0, write=20.0, pool=20.0)
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=timeout) as client:
        start = time.perf_counter()
        results = await asyncio.gather(
            *[_worker(client, task_id, requests_per_worker) for _ in range(concurrency)]
        )
        duration = time.perf_counter() - start

    all_latencies = []
    total_ok = 0
    total_errors = 0
    for r in results:
        total_ok += r["ok"]
        total_errors += r["errors"]
        all_latencies.extend(r["latencies_ms"])

    total_requests = concurrency * requests_per_worker
    success_rate = (total_ok / total_requests) * 100.0 if total_requests else 0.0
    throughput = total_ok / duration if duration > 0 else 0.0

    return {
        "task_id": task_id,
        "concurrency": concurrency,
        "requests_per_worker": requests_per_worker,
        "total_requests": total_requests,
        "successful_request_groups": total_ok,
        "failed_request_groups": total_errors,
        "success_rate_percent": round(success_rate, 2),
        "duration_sec": round(duration, 3),
        "throughput_groups_per_sec": round(throughput, 2),
        "latency_ms": {
            "mean": round(statistics.mean(all_latencies), 2) if all_latencies else 0.0,
            "p50": round(_percentile(all_latencies, 50), 2),
            "p95": round(_percentile(all_latencies, 95), 2),
            "p99": round(_percentile(all_latencies, 99), 2),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Concurrent API stress test for CloudScalerEnv")
    parser.add_argument("--task", default="medium-traffic-spike")
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--requests-per-worker", type=int, default=20)
    parser.add_argument("--output", default="stress_report.json")
    args = parser.parse_args()

    server = subprocess.Popen(["python", "-m", "src.app"], cwd=str(ROOT))
    try:
        time.sleep(2.5)
        report = asyncio.run(run_stress(args.task, args.concurrency, args.requests_per_worker))
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")

        print("Stress test complete")
        print(json.dumps(report, indent=2))

        if report["success_rate_percent"] < 99.0:
            print("WARNING: success rate below 99%")
            return 1
        return 0
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except Exception:
            server.kill()


if __name__ == "__main__":
    raise SystemExit(main())

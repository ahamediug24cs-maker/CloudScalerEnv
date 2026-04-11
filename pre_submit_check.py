import os
import re
import subprocess
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
INFERENCE = ROOT / "inference.py"

START_RE = re.compile(r"^\[START\] task=\S+ env=\S+ model=\S+$")
STEP_RE = re.compile(
    r"^\[STEP\] step=\d+ action=\S+ reward=-?\d+\.\d{2} done=(true|false) error=(null|.*)$"
)
END_RE = re.compile(
    r"^\[END\] success=(true|false) steps=\d+ score=-?\d+\.\d{2} rewards=(-?\d+\.\d{2})(,-?\d+\.\d{2})*$"
)


def check_file_contract() -> list[str]:
    issues: list[str] = []
    if not INFERENCE.exists():
        issues.append("inference.py missing at repo root")
        return issues

    content = INFERENCE.read_text(encoding="utf-8")
    if "from openai import OpenAI" not in content:
        issues.append("OpenAI client import not found in inference.py")
    if not re.search(r'API_BASE_URL\s*=\s*os\.getenv\(\s*"API_BASE_URL"\s*,', content):
        issues.append("API_BASE_URL default is missing")
    if not re.search(r'MODEL_NAME\s*=\s*os\.getenv\(\s*"MODEL_NAME"\s*,', content):
        issues.append("MODEL_NAME default is missing")
    if 'HF_TOKEN = os.getenv("HF_TOKEN")' not in content:
        issues.append("HF_TOKEN env var read not found")
    if "HF_TOKEN environment variable is required" not in content:
        issues.append("HF_TOKEN required validation message not found")
    return issues


def check_runtime_output() -> list[str]:
    issues: list[str] = []

    env = os.environ.copy()
    env.setdefault("HF_TOKEN", "dummy-token")
    env.setdefault("API_BASE_URL", "https://router.huggingface.co/v1")
    env.setdefault("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
    env.setdefault("TASK_NAME", "easy-memory-leak")

    proc = subprocess.run(
        ["python", "inference.py"],
        cwd=str(ROOT),
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
    )

    lines = [ln.rstrip("\n") for ln in proc.stdout.splitlines() if ln.strip()]
    if not lines:
        return ["No stdout lines produced by inference.py"]

    if not START_RE.match(lines[0]):
        issues.append("First line is not valid [START]")

    if not END_RE.match(lines[-1]):
        issues.append("Last line is not valid [END] with 2-decimal rewards")

    for line in lines[1:-1]:
        if not STEP_RE.match(line):
            issues.append(f"Invalid [STEP] line: {line}")
            break

    if any("\n" in ln or "\r" in ln for ln in lines):
        issues.append("Found newline/carriage return inside output line")

    return issues


def _http_call(method: str, url: str, payload: str | None = None) -> tuple[int, str]:
    data = payload.encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    req = Request(url, data=data, method=method, headers=headers)
    try:
        with urlopen(req, timeout=12) as resp:
            return resp.getcode(), resp.read().decode("utf-8", errors="replace")
    except URLError as exc:
        return 0, str(exc)


def check_api_smoke() -> list[str]:
    issues: list[str] = []
    server = subprocess.Popen(
        ["python", "-m", "src.app"],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        # Wait for server startup with retry loop
        base = "http://127.0.0.1:7860"
        max_retries = 10
        for attempt in range(max_retries):
            try:
                time.sleep(1)
                health_code, _ = _http_call("GET", f"{base}/health")
                if health_code == 200:
                    break
            except Exception:
                if attempt == max_retries - 1:
                    issues.append("Server failed to start after 10 seconds")
                    return issues

        reset_code, _ = _http_call("POST", f"{base}/reset", '{"task_id":"easy-memory-leak"}')
        step_code, _ = _http_call("POST", f"{base}/step", '{"action_type":"do_nothing"}')
        state_code, _ = _http_call("GET", f"{base}/state")

        if health_code != 200:
            issues.append(f"GET /health returned {health_code}")
        if reset_code != 200:
            issues.append(f"POST /reset returned {reset_code}")
        if step_code != 200:
            issues.append(f"POST /step returned {step_code}")
        if state_code != 200:
            issues.append(f"GET /state returned {state_code}")

        return issues
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except Exception:
            server.kill()


def main() -> int:
    issues = []
    issues.extend(check_file_contract())
    issues.extend(check_runtime_output())
    issues.extend(check_api_smoke())

    if issues:
        print("Pre-submit check: FAIL")
        for item in issues:
            print(f"- {item}")
        return 1

    print("Pre-submit check: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

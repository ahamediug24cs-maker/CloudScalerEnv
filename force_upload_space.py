from __future__ import annotations

import os
import sys
import time
import json
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

try:
    from huggingface_hub import HfApi
except ImportError as exc:
    raise SystemExit(
        "huggingface_hub is not installed. Run: pip install huggingface_hub"
    ) from exc


SPACE_REPO_ID = "Al-Ameen26/CloudScalerEnv"
SPACE_URL = "https://al-ameen26-cloudscalerenv.hf.space"
ROOT = Path(__file__).resolve().parent

REQUIRED_ROOT_FILES = [
    "Dockerfile",
    "inference.py",
    "openenv.yaml",
    "requirements.txt",
    "README.md",
    "pyproject.toml",
    "uv.lock",
]


def assert_required_files() -> None:
    missing = [name for name in REQUIRED_ROOT_FILES if not (ROOT / name).exists()]
    if not (ROOT / "src").exists():
        missing.append("src/")
    if not (ROOT / "server").exists():
        missing.append("server/")

    if missing:
        raise SystemExit("Missing required files: " + ", ".join(missing))


def http_call(method: str, path: str, payload: dict | None = None) -> tuple[int, str]:
    body = None
    headers = {}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = Request(f"{SPACE_URL}{path}", data=body, method=method, headers=headers)
    try:
        with urlopen(req, timeout=30) as resp:
            return resp.getcode(), resp.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")
    except URLError as exc:
        return 0, str(exc)


def main() -> int:
    token = os.getenv("HF_TOKEN")
    if not token:
        print("HF_TOKEN is not set.")
        print("Set it in PowerShell:")
        print('$env:HF_TOKEN="hf_xxx"')
        return 1

    assert_required_files()

    api = HfApi(token=token)

    # Upload only the files needed for evaluation and runtime.
    api.upload_folder(
        repo_id=SPACE_REPO_ID,
        repo_type="space",
        folder_path=str(ROOT),
        path_in_repo=".",
        allow_patterns=[
            "Dockerfile",
            "inference.py",
            "openenv.yaml",
            "requirements.txt",
            "README.md",
            "pyproject.toml",
            "uv.lock",
            "src/**",
            "server/**",
        ],
        ignore_patterns=[
            ".venv/**",
            "**/__pycache__/**",
            "**/*.pyc",
            ".git/**",
        ],
        commit_message="Force sync eval files and OpenEnv API",
    )

    print("Upload complete. Waiting 45s for Space restart...")
    time.sleep(45)

    code_root, body_root = http_call("GET", "/")
    print("GET / ->", code_root)
    print(body_root[:400])

    code_reset, body_reset = http_call("POST", "/reset", {})
    print("POST /reset ->", code_reset)
    print(body_reset[:400])

    if code_reset == 200:
        print("PASS: /reset is live and returns 200.")
        print("You can re-run the hackathon evaluation now.")
        return 0

    print("FAIL: /reset is still not 200. Check Space Files and build logs.")
    return 2


if __name__ == "__main__":
    sys.exit(main())

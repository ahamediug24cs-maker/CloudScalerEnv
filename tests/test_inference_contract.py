import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

START_RE = re.compile(r"^\[START\] task=\S+ env=\S+ model=\S+$")
STEP_RE = re.compile(r"^\[STEP\] step=\d+ action=\S+ reward=-?\d+\.\d{2} done=(true|false) error=(null|.*)$")
END_RE = re.compile(
    r"^\[END\] success=(true|false) steps=\d+ score=-?\d+\.\d{2} rewards=(-?\d+\.\d{2})(,-?\d+\.\d{2})*$"
)


def test_inference_output_shape_with_dummy_token() -> None:
    env = os.environ.copy()
    env["HF_TOKEN"] = "dummy-token"
    env["API_BASE_URL"] = "http://127.0.0.1:1/v1"
    env["MODEL_NAME"] = "test-model"
    env["TASK_NAME"] = "easy-memory-leak"

    proc = subprocess.run(["python", "inference.py"], cwd=str(ROOT), env=env, text=True, capture_output=True, timeout=60)
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]

    assert len(lines) >= 2
    assert START_RE.match(lines[0])
    assert END_RE.match(lines[-1])

    for line in lines[1:-1]:
        assert STEP_RE.match(line)


def test_inference_missing_token_still_emits_end() -> None:
    env = os.environ.copy()
    env.pop("HF_TOKEN", None)
    env["API_BASE_URL"] = "http://127.0.0.1:1/v1"
    env["TASK_NAME"] = "easy-memory-leak"

    proc = subprocess.run(["python", "inference.py"], cwd=str(ROOT), env=env, text=True, capture_output=True, timeout=60)
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]

    assert len(lines) >= 2
    assert START_RE.match(lines[0])
    assert END_RE.match(lines[-1])

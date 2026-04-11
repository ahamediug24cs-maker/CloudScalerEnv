import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _run(cmd: list[str]) -> tuple[int, str, str]:
    # Force UTF-8 for child Python processes on Windows to avoid cp1252 print crashes.
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True, env=env)
    return proc.returncode, proc.stdout, proc.stderr


def main() -> int:
    summary = {
        "pre_submit_check": {"ok": False},
        "validate_local": {"ok": False, "scores": {}},
        "baseline": {"ok": False, "scores": {}, "mean": None},
    }

    code, out, err = _run(["python", "pre_submit_check.py"])
    summary["pre_submit_check"]["ok"] = (code == 0 and "PASS" in out)

    code, out, err = _run(["python", "validate_local.py"])
    summary["validate_local"]["ok"] = (code == 0 and "Result: 5/5 checks passed" in (out + err))
    for line in out.splitlines():
        m = re.search(r"(easy|medium|hard): task loaded, grader score ([0-9.]+)", line)
        if m:
            summary["validate_local"]["scores"][m.group(1)] = float(m.group(2))

    code, out, err = _run(["python", "-m", "src.baseline", "--mode", "heuristic"])
    summary["baseline"]["ok"] = (code == 0)
    for line in out.splitlines():
        m = re.search(r"Task: (Easy|Medium|Hard) .*: ([0-9.]+)", line)
        if m:
            summary["baseline"]["scores"][m.group(1).lower()] = float(m.group(2))
        mm = re.search(r"Overall Mean Score: ([0-9.]+)", line)
        if mm:
            summary["baseline"]["mean"] = float(mm.group(1))

    all_ok = all(
        [
            summary["pre_submit_check"]["ok"],
            summary["validate_local"]["ok"],
            summary["baseline"]["ok"],
        ]
    )

    print(json.dumps(summary, indent=2, sort_keys=True))
    print("\nDRY_RUN_STATUS:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

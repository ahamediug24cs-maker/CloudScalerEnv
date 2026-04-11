import subprocess


def main() -> int:
    completed = subprocess.run(["python", "pre_submit_check.py"], text=True)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())

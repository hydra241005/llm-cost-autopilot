"""Cross-platform task runner.

Same targets as the Makefile, for environments without ``make`` — which on
Windows is most of them. Kept as a thin dispatch table so the two never drift
into meaning different things.

Usage::

    uv run python scripts/task.py check
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: Target name -> the commands it runs, in order.
TASKS: dict[str, list[list[str]]] = {
    "install": [
        ["uv", "venv", "--python", "3.11"],
        ["uv", "pip", "install", "-e", ".[dev,ml,db]"],
    ],
    "dev": [
        ["uv", "run", "uvicorn", "autopilot.api.main:get_app", "--factory", "--reload"],
    ],
    "test": [["uv", "run", "pytest"]],
    "lint": [["uv", "run", "ruff", "check", "."]],
    "typecheck": [["uv", "run", "mypy"]],
    "check": [
        ["uv", "run", "ruff", "check", "."],
        ["uv", "run", "mypy"],
        ["uv", "run", "pytest"],
    ],
    "baseline": [["uv", "run", "python", "scripts/baseline_run.py"]],
    "doctor": [["uv", "run", "python", "scripts/doctor.py"]],
}

#: Directories removed by ``clean``.
CLEAN_PATHS = (
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "htmlcov",
    ".coverage",
    "build",
    "dist",
)


def clean() -> int:
    """Remove caches and build artifacts."""
    for name in CLEAN_PATHS:
        target = PROJECT_ROOT / name
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
        elif target.exists():
            target.unlink()
    print("Removed caches and build artifacts.")
    return 0


def run_task(name: str) -> int:
    """Run every command for ``name``, stopping at the first failure.

    Args:
        name: Target to run.

    Returns:
        The exit code of the first failing command, or zero.
    """
    if name == "clean":
        return clean()
    for command in TASKS[name]:
        print(f"$ {' '.join(command)}", flush=True)
        result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
        if result.returncode != 0:
            return result.returncode
    return 0


def main() -> int:
    """Parse arguments and dispatch to the requested target."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("task", choices=[*sorted(TASKS), "clean"], help="Target to run.")
    args = parser.parse_args()
    return run_task(args.task)


if __name__ == "__main__":
    sys.exit(main())

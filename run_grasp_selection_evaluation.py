"""Run perception and grasp selection, then stop before robot execution."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
MAIN_SCRIPT = SCRIPT_DIR / "run_closed_loop.py"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the main pipeline in grasp-selection-only mode."
    )
    parser.add_argument("--scene", default="1")
    parser.add_argument("--case", default=None)
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--grasp-angle-weight", type=float, default=None)
    parser.add_argument("--grasp-angle-sigma-deg", type=float, default=None)
    parser.add_argument("--grasp-preferred-sigma-m", type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    command = [
        sys.executable,
        str(MAIN_SCRIPT),
        "--scene", str(args.scene),
        "--grasp-selection-only",
    ]
    if args.case is not None:
        command.extend(["--case", str(args.case)])
    for name, value in (
        ("--max-attempts", args.max_attempts),
        ("--grasp-angle-weight", args.grasp_angle_weight),
        ("--grasp-angle-sigma-deg", args.grasp_angle_sigma_deg),
        ("--grasp-preferred-sigma-m", args.grasp_preferred_sigma_m),
    ):
        if value is not None:
            command.extend([name, str(value)])
    raise SystemExit(subprocess.call(command, cwd=str(SCRIPT_DIR)))


if __name__ == "__main__":
    main()

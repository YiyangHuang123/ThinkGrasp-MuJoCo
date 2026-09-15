"""Run paired end-to-end comparisons for the full and baseline policies."""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
RUNNER = SCRIPT_DIR / "run_closed_loop.py"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "end_to_end_evaluation"


def parse_result(stdout: str, returncode: int) -> tuple[bool, int | None]:
    success_matches = re.findall(r"Task success:\s*(True|False)", stdout)
    step_matches = re.findall(r"Step count:\s*(\d+)", stdout)
    success = success_matches[-1] == "True" if success_matches else returncode == 0 and "task completed" in stdout.lower()
    steps = int(step_matches[-1]) if step_matches else None
    return success, steps


def snapshot_files(source: Path) -> dict[str, tuple[int, int]]:
    if not source.exists():
        return {}
    return {
        str(path.relative_to(source)): (path.stat().st_mtime_ns, path.stat().st_size)
        for path in source.rglob("*")
        if path.is_file()
    }


def archive_outputs(source: Path, destination: Path, before: dict[str, tuple[int, int]]) -> None:
    if not source.exists():
        return
    destination.mkdir(parents=True, exist_ok=True)
    output_root = destination / "closed_loop_outputs"
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        relative = str(path.relative_to(source))
        signature = (path.stat().st_mtime_ns, path.stat().st_size)
        if before.get(relative) == signature:
            continue
        target = output_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def run_one(
    scene: str,
    case: str | None,
    mode: str,
    index: int,
    output_dir: Path,
    max_attempts: int | None,
    scene_state: Path | None = None,
) -> dict:
    run_dir = output_dir / mode / f"run_{index:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    output_source = SCRIPT_DIR / "closed_loop_outputs"
    before_outputs = snapshot_files(output_source)

    command = [
        sys.executable,
        "-u",
        str(RUNNER),
        "--scene",
        scene,
        "--evaluation-mode",
        mode,
    ]
    if case:
        command.extend(["--case", case])
    if max_attempts is not None:
        command.extend(["--max-attempts", str(max_attempts)])
    if mode == "full" and scene_state is not None:
        command.extend(["--save-scene-state", str(scene_state)])
    elif mode == "baseline" and scene_state is not None:
        command.extend(["--restore-scene-state", str(scene_state)])

    started = datetime.now().isoformat(timespec="seconds")
    print(f"[{mode}] starting run {index}...", flush=True)
    process = subprocess.Popen(
        command,
        cwd=SCRIPT_DIR,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    output_lines = []
    assert process.stdout is not None
    for line in process.stdout:
        output_lines.append(line)
        print(f"[{mode} run {index}] {line}", end="", flush=True)
    return_code = process.wait()
    stdout = "".join(output_lines)
    (run_dir / "console.log").write_text(stdout, encoding="utf-8")
    archive_outputs(output_source, run_dir, before_outputs)

    success, steps = parse_result(stdout, return_code)
    result = {
        "scene": scene,
        "mode": mode,
        "run": index,
        "success": int(success),
        "step_count": "" if steps is None else steps,
        "return_code": return_code,
        "started_at": started,
        "result_dir": str(run_dir),
    }
    print(
        f"[{mode}] run {index}: success={bool(success)}, "
        f"step_count={steps}, return_code={return_code}"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True, help="Scene number or scene name, e.g. scene11")
    parser.add_argument("--case", default=None, help="Optional case file passed to run_closed_loop.py")
    parser.add_argument("--count", type=int, default=30, help="Trials per mode")
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    if args.count <= 0:
        raise SystemExit("--count must be positive")
    if not RUNNER.exists():
        raise SystemExit(f"Missing runner: {RUNNER}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    state_dir = output_dir / "paired_scene_states"
    state_dir.mkdir(parents=True, exist_ok=True)
    for index in range(1, args.count + 1):
        scene_state = state_dir / f"scene_{index:03d}.npz"
        results.append(
            run_one(
                args.scene,
                args.case,
                "full",
                index,
                output_dir,
                args.max_attempts,
                scene_state,
            )
        )
        results.append(
            run_one(
                args.scene,
                args.case,
                "baseline",
                index,
                output_dir,
                args.max_attempts,
                scene_state,
            )
        )

    summary_path = output_dir / "end_to_end_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved summary: {summary_path}")
    for mode in ("full", "baseline"):
        group = [row for row in results if row["mode"] == mode]
        successes = sum(row["success"] for row in group)
        steps = [int(row["step_count"]) for row in group if row["step_count"] != ""]
        mean_steps = sum(steps) / len(steps) if steps else None
        print(
            f"{mode}: {successes}/{len(group)} success "
            f"({100.0 * successes / len(group):.2f}%), "
            f"mean_step_count={mean_steps if mean_steps is not None else 'n/a'}"
        )


if __name__ == "__main__":
    main()

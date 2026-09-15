"""Run the grasp-selection-only experiment for multiple independent rounds."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SELECTION_SCRIPT = SCRIPT_DIR / "run_grasp_selection_evaluation.py"
GRASP_DEBUG_DIR = SCRIPT_DIR / "closed_loop_outputs" / "grasp_debug"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run grasp selection repeatedly and archive grasp_debug after "
            "each round."
        )
    )
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--scene", default="scene11")
    parser.add_argument("--case", default=None)
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=Path("grasp_selection_evaluation"),
    )
    return parser.parse_args()


def archive_grasp_debug(archive_root: Path, round_id: int):
    archive_path = archive_root / f"round_{round_id:03d}" / "grasp_debug"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists():
        shutil.rmtree(archive_path)
    if not GRASP_DEBUG_DIR.exists():
        print("No grasp_debug directory was produced in this round.", flush=True)
        return None
    shutil.copytree(GRASP_DEBUG_DIR, archive_path)
    print(f"Archived grasp_debug to: {archive_path}", flush=True)
    return archive_path


def append_summary(summary_path: Path, round_id: int, archive_path: Path):
    result_path = archive_path / "grasp_selection_result.json"
    if not result_path.exists():
        return
    result = json.loads(result_path.read_text(encoding="utf-8"))
    geometry = result["geometry_only"]
    guided = result["vlm_guided"]
    vlm_only = result["vlm_only"]
    row = {
        "round": round_id,
        "target_object": result.get("target_object", ""),
        "candidate_count": result.get("candidate_count", ""),
        "same_candidate": result.get("same_candidate", ""),
        "geometry_index": geometry.get("candidate_index", ""),
        "geometry_center_xyz": geometry.get("center_xyz", ""),
        "geometry_pose_xyzw": geometry.get("grasp_pose_xyzw", ""),
        "geometry_angle_deg": geometry.get("approach_angle_deg", ""),
        "geometry_angle_score": geometry.get("angle_score", ""),
        "guided_index": guided.get("candidate_index", ""),
        "guided_center_xyz": guided.get("center_xyz", ""),
        "guided_pose_xyzw": guided.get("grasp_pose_xyzw", ""),
        "guided_angle_deg": guided.get("approach_angle_deg", ""),
        "guided_preferred_score": guided.get("preferred_score", ""),
        "guided_final_score": guided.get("final_score", ""),
        "vlm_only_index": vlm_only.get("candidate_index", ""),
        "vlm_only_center_xyz": vlm_only.get("center_xyz", ""),
        "vlm_only_pose_xyzw": vlm_only.get("grasp_pose_xyzw", ""),
        "vlm_only_angle_deg": vlm_only.get("approach_angle_deg", ""),
        "vlm_only_preferred_score": vlm_only.get("preferred_score", ""),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not summary_path.exists()
    with summary_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()
    if args.count < 1:
        raise ValueError("--count must be positive")
    if not SELECTION_SCRIPT.exists():
        raise FileNotFoundError(f"Missing selection script: {SELECTION_SCRIPT}")

    archive_root = args.archive_dir.expanduser().resolve()
    success_count = 0
    failed_count = 0
    archived_count = 0

    for round_id in range(1, args.count + 1):
        print(
            f"\n========== Grasp-selection round "
            f"{round_id}/{args.count} ==========",
            flush=True,
        )
        command = [
            sys.executable,
            str(SELECTION_SCRIPT),
            "--scene",
            str(args.scene),
        ]
        if args.case is not None:
            command.extend(["--case", str(args.case)])

        try:
            subprocess.run(command, cwd=str(SCRIPT_DIR), check=True)
        except subprocess.CalledProcessError as exc:
            failed_count += 1
            print(
                f"Round {round_id} failed with exit code {exc.returncode}.",
                flush=True,
            )
        else:
            success_count += 1
            print(f"Round {round_id} completed successfully.", flush=True)
        finally:
            archive_path = archive_grasp_debug(archive_root, round_id)
            if archive_path is not None:
                archived_count += 1
                append_summary(archive_root / "grasp_selection_summary.csv", round_id, archive_path)

    print(
        f"\nFinished {args.count} rounds: "
        f"successful={success_count}, failed={failed_count}, "
        f"archived={archived_count}",
        flush=True,
    )


if __name__ == "__main__":
    main()

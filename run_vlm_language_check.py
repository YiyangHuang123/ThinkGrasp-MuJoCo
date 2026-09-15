"""VLM-only language-understanding pilot runner.

This script evaluates whether the VLM can map intent-based language
instructions to the expected target object. It does not run GroundingDINO,
GraspNet, or Panda execution.

Expected location on the server:
    project_root/run_vlm_language_check.py
    project_root/vlm_bridge.py
    project_root/perception_viz.py
    project_root/vlm_system_prompt.txt
    project_root/cases/vlm_language_cases/*.txt

Each case file uses the same two-line format as run_closed_loop.py:
    line 1: language instruction
    line 2: expected target object

Images:
    By default, the script looks for an image with the same stem as the case:
        cases/vlm_language_cases/images/case_A01_take_photo.png

    For a quick pilot, a shared image can also be passed with --image.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from vlm_bridge import run_vlm_selection
from perception_viz import save_vlm_selection_visualization


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CASE_DIR = SCRIPT_DIR / "cases" / "vlm_language_cases"
DEFAULT_PROMPT_PATH = SCRIPT_DIR / "vlm_system_prompt.txt"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "vlm_capability_outputs" / "language"


def load_case(case_path: Path) -> dict:
    lines = case_path.read_text(encoding="utf-8").splitlines()

    if len(lines) < 2:
        raise ValueError(
            f"Case file must contain at least two lines: {case_path}"
        )

    instruction = lines[0].strip()
    expected_target = lines[1].strip()

    if not instruction or not expected_target:
        raise ValueError(
            f"Case file has empty instruction or target: {case_path}"
        )

    return {
        "case_id": case_path.stem,
        "case_path": str(case_path),
        "instruction": instruction,
        "expected_target": expected_target,
    }


def normalize_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def resolve_case_image(case_path: Path, shared_image: Path | None) -> Path:
    if shared_image is not None:
        return shared_image

    image_dir = case_path.parent / "images"
    for suffix in (".png", ".jpg", ".jpeg"):
        image_path = image_dir / f"{case_path.stem}{suffix}"
        if image_path.is_file():
            return image_path

    raise FileNotFoundError(
        "No image found for case. Expected one of: "
        f"{image_dir / (case_path.stem + '.png')}, "
        f"{image_dir / (case_path.stem + '.jpg')}, or pass --image."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run VLM-only language understanding checks."
    )
    parser.add_argument(
        "--case-dir",
        type=Path,
        default=DEFAULT_CASE_DIR,
        help="Directory containing two-line VLM language case files.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Optional shared top-view image used for all cases.",
    )
    parser.add_argument(
        "--system-prompt",
        type=Path,
        default=DEFAULT_PROMPT_PATH,
        help="VLM system prompt path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for CSV, JSON, and visualization outputs.",
    )
    args = parser.parse_args()

    case_dir = args.case_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    visual_dir = output_dir / "visualizations"
    json_dir = output_dir / "json"

    output_dir.mkdir(parents=True, exist_ok=True)
    visual_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    shared_image = None
    if args.image is not None:
        shared_image = args.image.expanduser().resolve()
        if not shared_image.is_file():
            raise FileNotFoundError(f"Shared image does not exist: {shared_image}")

    system_prompt = args.system_prompt.expanduser().resolve()
    if not system_prompt.is_file():
        raise FileNotFoundError(f"System prompt does not exist: {system_prompt}")

    case_paths = sorted(case_dir.glob("*.txt"))
    if not case_paths:
        raise FileNotFoundError(f"No .txt case files found in {case_dir}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"vlm_language_results_{timestamp}.csv"

    rows = []

    for case_path in case_paths:
        case = load_case(case_path)
        image_path = resolve_case_image(case_path, shared_image)

        print()
        print(f"Running {case['case_id']}")
        print("Instruction:", case["instruction"])
        print("Expected target:", case["expected_target"])
        print("Image:", image_path)

        vlm_result = run_vlm_selection(
            image_path=image_path,
            goal=case["instruction"],
            system_prompt_path=system_prompt,
        )

        selected_object = str(vlm_result["selected_object"]).strip()
        preferred_location = int(vlm_result["preferred_grasping_location"])

        correct = (
            normalize_name(selected_object)
            == normalize_name(case["expected_target"])
        )

        image = np.asarray(Image.open(image_path).convert("RGB"))
        viz_path = visual_dir / f"{case['case_id']}_vlm.png"
        save_vlm_selection_visualization(
            image=image,
            bbox_xyxy=vlm_result["result"]["cropping_box"],
            selected_object=selected_object,
            preferred_location=preferred_location,
            centroid_xy=vlm_result["selected_properties"]["centroid_coordinates"],
            show_info_box=False,
            output_path=viz_path,
        )

        json_path = json_dir / f"{case['case_id']}.json"
        payload = {
            **case,
            "image_path": str(image_path),
            "selected_object": selected_object,
            "preferred_grasping_location": preferred_location,
            "correct": bool(correct),
            "visualization_path": str(viz_path),
            "raw_output": vlm_result.get("raw_output"),
            "vlm_result": vlm_result,
        }
        json_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        row = {
            "case_id": case["case_id"],
            "instruction": case["instruction"],
            "expected_target": case["expected_target"],
            "selected_object": selected_object,
            "preferred_grasping_location": preferred_location,
            "correct": int(correct),
            "image_path": str(image_path),
            "visualization_path": str(viz_path),
            "json_path": str(json_path),
        }
        rows.append(row)

        print("Selected object:", selected_object)
        print("Preferred grasping location:", preferred_location)
        print("Correct:", correct)

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    correct_count = sum(row["correct"] for row in rows)
    total_count = len(rows)
    accuracy = correct_count / total_count if total_count else 0.0

    summary_path = output_dir / f"vlm_language_summary_{timestamp}.json"
    summary_path.write_text(
        json.dumps(
            {
                "total_cases": total_count,
                "correct_cases": int(correct_count),
                "accuracy": accuracy,
                "csv_path": str(csv_path),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("VLM language check complete.")
    print("Correct / total:", f"{correct_count}/{total_count}")
    print("Accuracy:", f"{accuracy:.3f}")
    print("CSV saved:", csv_path)
    print("Summary saved:", summary_path)


if __name__ == "__main__":
    main()

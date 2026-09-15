import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from robosuite.controllers import load_composite_controller_config
from thinkgrasp_minimal_env import ThinkGraspMinimalEnv
from scene_bridge import FIXED_PERCEPTION_CROP_XYXY


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return str(value)


def collect_case(output_dir, index):
    controller_config = load_composite_controller_config(controller="BASIC")
    env = ThinkGraspMinimalEnv(
        controller_configs=controller_config,
        scene_name="scene11",
    )
    try:
        rgb, _, _ = env.render_camera(
            {
                "camera_name": "topview",
                "width": 640,
                "height": 640,
            }
        )
        crop_x1, crop_y1, crop_x2, crop_y2 = [int(value) for value in FIXED_PERCEPTION_CROP_XYXY]
        workspace_rgb = np.asarray(rgb)[crop_y1:crop_y2, crop_x1:crop_x2].copy()

        case_id = f"scene11_{index:03d}"
        image_path = output_dir / "scenes" / f"{case_id}.png"
        metadata_path = output_dir / "metadata" / f"{case_id}.json"
        imageio.imwrite(image_path, workspace_rgb)
        metadata = {
            "case_id": case_id,
            "scene_name": "scene11",
            "image": str(image_path),
            "objects": env.get_object_poses(),
        }
        metadata_path.write_text(
            json.dumps(metadata, indent=2, default=json_default),
            encoding="utf-8",
        )
        print(f"Saved {image_path}")
        print(f"Saved {metadata_path}")
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("localization_evaluation"),
    )
    args = parser.parse_args()
    if args.count < 1:
        raise ValueError("--count must be positive")
    (args.output_dir / "scenes").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metadata").mkdir(parents=True, exist_ok=True)
    skipped = 0
    for index in range(1, args.count + 1):
        print(f"Generating scene11 case {index}/{args.count}")
        try:
            collect_case(args.output_dir, index)
        except RuntimeError as exc:
            if not str(exc).startswith(
                "Failed to generate a valid GSO clutter scene after "
            ):
                raise
            skipped += 1
            print(f"Skipped scene11_{index:03d}: {exc}", flush=True)
    print(f"Finished: saved={args.count - skipped}, skipped={skipped}")


if __name__ == "__main__":
    main()



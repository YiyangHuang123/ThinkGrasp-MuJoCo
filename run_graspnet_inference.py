"""GraspNet inference worker for the ThinkGrasp-MuJoCo bridge.

This script is executed with the legacy ThinkGrasp / GraspNet Python
environment. It receives a MuJoCo world-frame point cloud and returns
assigned ThinkGrasp 7D grasp poses.

Example:
    python run_graspnet_inference.py \
        --input bridge_data/mujoco_scene_pointcloud.npz \
        --output bridge_data/thinkgrasp_assigned_grasps.npz
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import open3d as o3d


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR

# Keep the MuJoCo migration prototype self-contained: prefer the local
# grasp_detector.py. The original ThinkGrasp root remains available only for
# its GraspNet model / utils dependencies imported by that detector.
if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from grasp_detector import Graspnet


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run GraspNet on a MuJoCo scene point cloud."
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input .npz containing points, colors and object poses.",
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output .npz containing assigned 7D grasp poses.",
    )


    return parser.parse_args()


def main():
    args = parse_args()

    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input point-cloud file does not exist: {input_path}"
        )

    data = np.load(input_path)

    required_keys = {
        "points",
        "colors",
    }

    missing_keys = required_keys.difference(data.files)

    if missing_keys:
        raise KeyError(
            f"Input file is missing keys: {sorted(missing_keys)}"
        )

    points = data["points"].astype(np.float32)
    colors = data["colors"].astype(np.float32)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"points must have shape (N, 3), got {points.shape}"
        )

    if colors.shape != points.shape:
        raise ValueError(
            f"colors shape {colors.shape} does not match "
            f"points shape {points.shape}"
        )

    if not np.isfinite(points).all():
        raise ValueError("Input point cloud contains NaN or Inf.")

    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(points)
    pointcloud.colors = o3d.utility.Vector3dVector(colors)


    graspnet = Graspnet()

    grasp_group = graspnet.compute_grasp_pose(
        pointcloud
    )


    # Keep GraspNet's original collision detection. Disable later
    # MuJoCo-specific hard angle / height deletion so the MuJoCo main
    # process can reproduce the original PyBullet object-assignment and
    # 15-degree soft-preference logic itself.
    (
        target_poses,
        _,
        target_angles,
        target_scores,
    ) = graspnet.filter_grasp_pose_for_target(
        grasp_group,
        max_approach_angle_deg=180.0,
        min_grasp_center_z=-np.inf,
    )

    target_pose_array = np.asarray(
        target_poses,
        dtype=np.float64,
    ).reshape(-1, 7)

    diagnostics = getattr(
        graspnet,
        "last_filter_diagnostics",
        None,
    )

    if diagnostics is None:
        raise RuntimeError(
            "Grasp detector did not expose orientation diagnostics."
        )

    raw_rotation_matrices = np.asarray(
        diagnostics["raw_rotation_matrices"],
        dtype=np.float64,
    ).reshape(-1, 3, 3)
    converted_rotation_matrices = np.asarray(
        diagnostics["converted_rotation_matrices"],
        dtype=np.float64,
    ).reshape(-1, 3, 3)
    diagnostic_centers = np.asarray(
        diagnostics["centers"],
        dtype=np.float64,
    ).reshape(-1, 3)
    diagnostic_source_indices = np.asarray(
        diagnostics["source_indices"],
        dtype=np.int64,
    ).reshape(-1)

    if not (
        len(target_pose_array)
        == len(raw_rotation_matrices)
        == len(converted_rotation_matrices)
        == len(diagnostic_centers)
        == len(diagnostic_source_indices)
    ):
        raise RuntimeError(
            "Grasp orientation diagnostic arrays are not aligned "
            "with returned 7D grasp poses."
        )

    output_data = {
        "grasps": target_pose_array,
        "grasp_angles_deg": target_angles,
        "grasp_scores": target_scores,
        "raw_graspnet_rotation_matrices": raw_rotation_matrices,
        "converted_rotation_matrices": converted_rotation_matrices,
        "diagnostic_grasp_centers": diagnostic_centers,
        "diagnostic_source_indices": diagnostic_source_indices,
    }


    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        output_path,
        **output_data,
    )

    print(
        "GRASPNET_SUMMARY "
        f"input_points={len(pointcloud.points)} "
        f"raw_grasps={len(grasp_group)} "
        f"collision_filtered={len(target_pose_array)}"
    )



if __name__ == "__main__":
    main()

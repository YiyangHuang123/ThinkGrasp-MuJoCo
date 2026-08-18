"""Run the ORIGINAL ThinkGrasp PyBullet point-cloud fusion on MuJoCo views.

This worker intentionally imports `process_pcds` and `reconstruction_config`
directly from the local `pybullet_pointcloud_fusion.py` source copy. Therefore the actual
outlier / normals / voxel / ICP / merge implementation and its parameters are
the source implementation rather than a rewritten approximation.

Only camera acquisition differs: MuJoCo has already transformed each camera
point cloud into the shared world frame before this worker is called.
"""

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


SCRIPT_DIR = Path(__file__).resolve().parent

from pybullet_pointcloud_fusion import (
    process_pcds,
    reconstruction_config,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    with np.load(
        input_path,
        allow_pickle=False,
    ) as data:
        camera_count = int(data["camera_count"])
        camera_names = [
            str(name)
            for name in data["camera_names"]
        ]

        pcds = []

        for camera_index in range(camera_count):
            points = data[
                f"points_{camera_index}"
            ].astype(np.float64)
            colors = data[
                f"colors_{camera_index}"
            ].astype(np.float64)

            if points.ndim != 2 or points.shape[1] != 3:
                raise ValueError(
                    f"points_{camera_index} must be Nx3, "
                    f"got {points.shape}"
                )

            if colors.shape != points.shape:
                raise ValueError(
                    f"colors_{camera_index} shape {colors.shape} "
                    f"does not match points {points.shape}"
                )

            if len(points) == 0:
                raise RuntimeError(
                    f"Camera {camera_index} point cloud is empty."
                )

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(
                points
            )
            pcd.colors = o3d.utility.Vector3dVector(
                colors
            )
            pcds.append(pcd)

            camera_name = (
                camera_names[camera_index]
                if camera_index < len(camera_names)
                else str(camera_index)
            )
            print(
                "Source fusion input camera:",
                camera_name,
                "points:",
                len(points),
            )

    if len(pcds) == 0:
        raise RuntimeError(
            "No camera point clouds were supplied."
        )

    print(
        "Using ORIGINAL ThinkGrasp reconstruction_config:",
        reconstruction_config,
    )
    print(
        "Calling local source copy pybullet_pointcloud_fusion.process_pcds()."
    )

    transformations, fused_pcd = process_pcds(
        pcds,
        reconstruction_config,
    )

    fused_points = np.asarray(
        fused_pcd.points,
        dtype=np.float32,
    )
    fused_colors = np.asarray(
        fused_pcd.colors,
        dtype=np.float32,
    )

    if len(fused_points) == 0:
        raise RuntimeError(
            "Original ThinkGrasp process_pcds() returned an empty cloud."
        )

    for camera_index in sorted(
        transformations.keys()
    ):
        transform = np.asarray(
            transformations[camera_index],
            dtype=np.float64,
        )
        print(
            f"ICP transform camera {camera_index}:"
        )
        print(transform)
        print(
            "  trace:",
            float(np.trace(transform)),
        )
        print(
            "  translation_norm:",
            float(
                np.linalg.norm(
                    transform[:3, 3]
                )
            ),
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        output_path,
        points=fused_points,
        colors=fused_colors,
        voxel_size=np.asarray(
            reconstruction_config["voxel_size"],
            dtype=np.float64,
        ),
        nb_neighbors=np.asarray(
            reconstruction_config["nb_neighbors"],
            dtype=np.int64,
        ),
        std_ratio=np.asarray(
            reconstruction_config["std_ratio"],
            dtype=np.float64,
        ),
        icp_max_try=np.asarray(
            reconstruction_config["icp_max_try"],
            dtype=np.int64,
        ),
        icp_max_iter=np.asarray(
            reconstruction_config["icp_max_iter"],
            dtype=np.int64,
        ),
        translation_thresh=np.asarray(
            reconstruction_config["translation_thresh"],
            dtype=np.float64,
        ),
        rotation_thresh=np.asarray(
            reconstruction_config["rotation_thresh"],
            dtype=np.float64,
        ),
        max_correspondence_distance=np.asarray(
            reconstruction_config[
                "max_correspondence_distance"
            ],
            dtype=np.float64,
        ),
    )

    print(
        "Source-fused output points:",
        len(fused_points),
    )
    print(
        "Source-fused output:",
        output_path,
    )


if __name__ == "__main__":
    main()

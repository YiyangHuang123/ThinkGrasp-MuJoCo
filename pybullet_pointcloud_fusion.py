"""Local copy of the original ThinkGrasp point-cloud fusion logic.

Source: original ThinkGrasp `utils.py`.

Only the point-cloud fusion configuration and `process_pcds()` routine are
copied here so the MuJoCo migration project is self-contained. The algorithm,
order of operations, thresholds, and Open3D / open3d_plus calls are preserved.
"""

import numpy as np
import open3d as o3d
import open3d_plus as o3dp


# Copied from original ThinkGrasp utils.py.
reconstruction_config = {
    "nb_neighbors": 50,
    "std_ratio": 2.0,
    "voxel_size": 0.0015,
    "icp_max_try": 5,
    "icp_max_iter": 2000,
    "translation_thresh": 3.95,
    "rotation_thresh": 0.02,
    "max_correspondence_distance": 0.02,
}


def process_pcds(pcds, reconstruction_config):
    """Copied from original ThinkGrasp utils.py."""

    trans = dict()
    pcd = pcds[0]
    pcd.estimate_normals()
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=reconstruction_config["nb_neighbors"],
        std_ratio=reconstruction_config["std_ratio"],
    )

    for i in range(1, len(pcds)):
        voxel_size = reconstruction_config["voxel_size"]

        income_pcd, _ = pcds[i].remove_statistical_outlier(
            nb_neighbors=reconstruction_config["nb_neighbors"],
            std_ratio=reconstruction_config["std_ratio"],
        )
        income_pcd.estimate_normals()
        income_pcd = income_pcd.voxel_down_sample(voxel_size)

        transok_flag = False

        for _ in range(reconstruction_config["icp_max_try"]):
            reg_p2p = o3d.pipelines.registration.registration_icp(
                income_pcd,
                pcd,
                reconstruction_config[
                    "max_correspondence_distance"
                ],
                np.eye(4, dtype=np.float64),
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    reconstruction_config["icp_max_iter"]
                ),
            )

            if (
                np.trace(reg_p2p.transformation)
                > reconstruction_config["translation_thresh"]
            ) and (
                np.linalg.norm(
                    reg_p2p.transformation[:3, 3]
                )
                < reconstruction_config["rotation_thresh"]
            ):
                transok_flag = True
                break

        if not transok_flag:
            reg_p2p.transformation = np.eye(
                4,
                dtype=np.float32,
            )

        income_pcd = income_pcd.transform(
            reg_p2p.transformation
        )
        trans[i] = reg_p2p.transformation

        pcd = o3dp.merge_pcds(
            [pcd, income_pcd]
        )
        pcd = pcd.voxel_down_sample(
            voxel_size
        )
        pcd.estimate_normals()

    return trans, pcd

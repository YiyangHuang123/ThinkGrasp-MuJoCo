import numpy as np
import open3d as o3d
import open3d_plus as o3dp
from scipy.spatial.transform import Rotation as R
import copy

from models.graspnet.graspnet_baseline import GraspNetBaseLine
from graspnet_config import graspnet_config


class Graspnet:
    def __init__(self):
        self.config = graspnet_config
        self.graspnet_baseline = GraspNetBaseLine(
            checkpoint_path=self.config["graspnet_checkpoint_path"]
        )

    def compute_grasp_pose(self, full_pcd):
        points, _ = o3dp.pcd2array(full_pcd)
        grasp_pcd = copy.deepcopy(full_pcd)
        grasp_pcd.points = o3d.utility.Vector3dVector(-points)

        gg = self.graspnet_baseline.inference(grasp_pcd)
        gg.translations = -gg.translations
        gg.rotation_matrices = -gg.rotation_matrices
        gg.translations = (
            gg.translations
            + gg.rotation_matrices[:, :, 0]
            * self.config["refine_approach_dist"]
        )
        gg = self.graspnet_baseline.collision_detection(
            gg,
            points,
        )

        return gg

    def filter_grasp_pose_for_target(
        self,
        gg,
        max_approach_angle_deg=45.0,
        min_grasp_center_z=0.830,
    ):
        """Filter and rank grasps for one GroundingDINO target crop.

        Ranking:
            1. reject approach angles greater than 45 degrees;
            2. sort remaining grasps by GraspNet score descending;
            3. use approach angle ascending as the tie-breaker.
        """

        if len(gg) == 0:
            print("Target crop: no raw grasp candidates")
            self.last_filter_diagnostics = {
                "source_indices": np.empty((0,), dtype=np.int64),
                "centers": np.empty((0, 3), dtype=np.float64),
                "raw_rotation_matrices": np.empty((0, 3, 3), dtype=np.float64),
                "converted_rotation_matrices": np.empty((0, 3, 3), dtype=np.float64),
            }
            return (
                [],
                [],
                np.empty((0,), dtype=np.float64),
                np.empty((0,), dtype=np.float64),
            )

        rs = gg.rotation_matrices
        ts = gg.translations
        depths = gg.depths
        scores = gg.scores

        ts = (
            ts
            + rs[:, :, 0]
            * np.vstack((depths, depths, depths)).T
        )

        # Preserve the RAW grasp semantics that were validated visually:
        #   RAW-X = approach direction
        #   RAW-Y = finger-opening direction
        #
        # Express the same physical grasp in the ThinkGrasp / Panda pose
        # convention used downstream:
        #   +Z = approach
        #   +X = finger-opening
        #
        # The RAW matrices here are left-handed because compute_grasp_pose()
        # negates all three axes. Therefore do not pass them through directly.
        # Rebuild a legal right-handed frame from the two physically meaningful
        # RAW directions instead.
        eelink_rs = np.zeros(
            shape=(len(rs), 3, 3),
            dtype=np.float64,
        )

        for rotation_index in range(len(rs)):
            approach = rs[
                rotation_index,
                :,
                0,
            ].astype(
                np.float64
            ).copy()

            opening = rs[
                rotation_index,
                :,
                1,
            ].astype(
                np.float64
            ).copy()

            approach /= max(
                np.linalg.norm(approach),
                1e-12,
            )

            # Remove tiny numerical non-orthogonality before rebuilding the
            # right-handed frame.
            opening = (
                opening
                - np.dot(
                    opening,
                    approach,
                )
                * approach
            )

            opening /= max(
                np.linalg.norm(opening),
                1e-12,
            )

            # For columns [X, Y, Z]:
            #   X = opening
            #   Z = approach
            # choose Y = Z x X so X x Y = Z.
            side = np.cross(
                approach,
                opening,
            )

            side /= max(
                np.linalg.norm(side),
                1e-12,
            )

            eelink_rs[
                rotation_index,
                :,
                0,
            ] = opening

            eelink_rs[
                rotation_index,
                :,
                1,
            ] = side

            eelink_rs[
                rotation_index,
                :,
                2,
            ] = approach

        approach_cosines = np.clip(
            -rs[:, 2, 0],
            -1.0,
            1.0,
        )

        approach_angles_deg = np.degrees(
            np.arccos(approach_cosines)
        )

        height_safe_mask = (
            ts[:, 2] >= float(min_grasp_center_z)
        )

        angle_safe_mask = (
            approach_angles_deg
            <= float(max_approach_angle_deg)
        )

        safe_indices = np.flatnonzero(
            height_safe_mask
            & angle_safe_mask
        )

        if len(safe_indices) == 0:
            self.last_filter_diagnostics = {
                "source_indices": np.empty((0,), dtype=np.int64),
                "centers": np.empty((0, 3), dtype=np.float64),
                "raw_rotation_matrices": np.empty((0, 3, 3), dtype=np.float64),
                "converted_rotation_matrices": np.empty((0, 3, 3), dtype=np.float64),
            }
            print(
                "Target crop: no grasp passed the safety filters; "
                f"required centre Z >= {min_grasp_center_z:.3f} m "
                f"and angle <= {max_approach_angle_deg:.1f} deg. "
                f"Raw centre Z range = "
                f"[{ts[:, 2].min():.3f}, {ts[:, 2].max():.3f}] m; "
                f"minimum raw angle = "
                f"{approach_angles_deg.min():.2f} deg"
            )
            return (
                [],
                [],
                np.empty((0,), dtype=np.float64),
                np.empty((0,), dtype=np.float64),
            )

        safe_angles = approach_angles_deg[safe_indices]
        safe_scores = scores[safe_indices]

        # np.lexsort uses the final key as the primary key.
        # Primary: GraspNet score descending.
        # Secondary: approach angle ascending.
        order = np.lexsort(
            (
                safe_angles,
                -safe_scores,
            )
        )

        sorted_indices = safe_indices[order]

        sorted_angles = approach_angles_deg[sorted_indices]
        sorted_scores = scores[sorted_indices]
        sorted_translations = ts[sorted_indices]
        sorted_rotations = eelink_rs[sorted_indices]
        sorted_raw_rotations = rs[sorted_indices].copy()
        sorted_gg = gg[sorted_indices]

        print(
            f"Target crop: kept {len(sorted_indices)} grasp(s) "
            f"with centre Z >= {min_grasp_center_z:.3f} m "
            f"and angle <= {max_approach_angle_deg:.1f} deg; "
            f"best centre Z = "
            f"{sorted_translations[0, 2]:.3f} m; "
            f"selected score = {sorted_scores[0]:.4f}; "
            f"selected angle = {sorted_angles[0]:.2f} deg"
        )

        grasp_poses = []
        grasp_geometries = []
        final_converted_rotations = []

        for grasp_index in range(len(sorted_gg)):
            grasp_geometries.append(
                sorted_gg[
                    grasp_index
                ].to_open3d_geometry()
            )

            grasp_rotation_matrix = (
                sorted_rotations[grasp_index].copy()
            )

            handedness_error = np.linalg.norm(
                np.cross(
                    grasp_rotation_matrix[:, 0],
                    grasp_rotation_matrix[:, 1],
                )
                - grasp_rotation_matrix[:, 2]
            )

            if handedness_error > 1e-5:
                raise RuntimeError(
                    "Rebuilt grasp rotation is not right-handed: "
                    f"error={handedness_error:.6e}"
                )

            grasp_pose = np.zeros(
                7,
                dtype=np.float64,
            )

            grasp_pose[:3] = sorted_translations[
                grasp_index
            ]

            rotation = R.from_matrix(
                grasp_rotation_matrix
            )
            grasp_pose[-4:] = rotation.as_quat()

            grasp_poses.append(grasp_pose)
            final_converted_rotations.append(
                grasp_rotation_matrix.copy()
            )

        self.last_filter_diagnostics = {
            "source_indices": sorted_indices.astype(np.int64),
            "centers": sorted_translations.astype(np.float64),
            "raw_rotation_matrices": sorted_raw_rotations.astype(np.float64),
            "converted_rotation_matrices": np.asarray(
                final_converted_rotations,
                dtype=np.float64,
            ).reshape(-1, 3, 3),
        }

        return (
            grasp_poses,
            grasp_geometries,
            sorted_angles.astype(np.float64),
            sorted_scores.astype(np.float64),
        )

    def grasp_detection(self, full_pcd):
        grasp_group = self.compute_grasp_pose(full_pcd)

        return self.filter_grasp_pose_for_target(
            grasp_group,
            max_approach_angle_deg=45.0,
        )

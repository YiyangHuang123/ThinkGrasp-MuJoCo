"""Minimal ThinkGrasp-style closed loop in MuJoCo.

Pipeline:
    fixed testcase (language goal + simulator GT target)
    -> MuJoCo perception
    -> GraspNet subprocess
    -> assign grasps to nearby objects and select by DINO preferred location
    -> Panda execution
    -> success check
    -> on failure: recover, then re-perceive and re-plan
       (one selected grasp per perception cycle, PyBullet-style)
"""

from pathlib import Path
from datetime import datetime
import argparse
import os
import subprocess
import sys
import traceback

import imageio.v2 as imageio
import numpy as np
from robosuite.controllers import load_composite_controller_config

from graspnet_bridge import (
    load_target_grasp_data,
    run_graspnet_inference,
)
from dual_view_recorder import DualViewRecorder
from scene_bridge import (
    FIXED_PERCEPTION_CROP_XYXY,
    build_pybullet_style_heightmap,
    export_colored_pointcloud_ply,
    export_pybullet_style_target_scene,
)
from thinkgrasp_minimal_env import (
    ThinkGraspMinimalEnv,
    load_thinkgrasp_joint_position_controller_config,
)
from vlm_bridge import run_vlm_selection
from perception_viz import (
    save_grounding_grid_visualization,
    save_vlm_selection_visualization,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR

# PyBullet-style fixed test cases.
# Line 1: natural-language goal for VLM.
# Line 2: simulator GT target object name, used ONLY for final success check.
CASE_DIR = SCRIPT_DIR / "cases"
DEFAULT_CASE_PATH = CASE_DIR / "case02_white_ramekin.txt"

BRIDGE_DIR = SCRIPT_DIR / "bridge_data"
SCENE_PATH = BRIDGE_DIR / "closed_loop_scene.npz"
GRASP_PATH = BRIDGE_DIR / "closed_loop_grasps.npz"

# Source-faithful fallback artifacts:
# target crop -> 0 grasps -> workspace-filtered full four-view scene.
FULL_SCENE_PATH = BRIDGE_DIR / "closed_loop_full_scene.npz"
FULL_SCENE_RAW_VIEWS_PATH = (
    BRIDGE_DIR / "closed_loop_full_scene_raw_views.npz"
)
FULL_GRASP_PATH = BRIDGE_DIR / "closed_loop_full_scene_grasps.npz"
FULL_TARGET_FILTERED_GRASP_PATH = (
    BRIDGE_DIR / "closed_loop_full_scene_target_filtered_grasps.npz"
)

GROUNDING_IMAGE_PATH = BRIDGE_DIR / "perception_rgb.png"
FULL_PERCEPTION_IMAGE_PATH = BRIDGE_DIR / "topview_full_rgb.png"
GROUNDING_RESULT_PATH = BRIDGE_DIR / "grounding_result.npz"
GROUNDING_VIS_PATH = BRIDGE_DIR / "grounding_bbox.png"

# Frozen project-local copy of the original ThinkGrasp VLM system prompt.
# This avoids a runtime dependency on ../simulation_main.py.
VLM_SYSTEM_PROMPT_PATH = SCRIPT_DIR / "vlm_system_prompt.txt"

# Human-readable closed-loop outputs, grouped by purpose.
CLOSED_LOOP_OUTPUT_DIR = (
    SCRIPT_DIR / "closed_loop_outputs"
)
VLM_SELECTION_OUTPUT_DIR = (
    CLOSED_LOOP_OUTPUT_DIR / "vlm_selection"
)
GROUNDING_GRID_OUTPUT_DIR = (
    CLOSED_LOOP_OUTPUT_DIR / "grounding_grid"
)
FUSED_CLOUD_PREVIEW_DIR = (
    CLOSED_LOOP_OUTPUT_DIR / "fused_clouds"
)
PERCEPTION_VIEW_OUTPUT_DIR = (
    CLOSED_LOOP_OUTPUT_DIR / "perception_views"
)
GRASP_DEBUG_PREVIEW_DIR = (
    CLOSED_LOOP_OUTPUT_DIR / "grasp_debug"
)
GRASP_DEBUG_ALL_GRASPS_DIR = (
    GRASP_DEBUG_PREVIEW_DIR / "all_grasps"
)
GRASP_DEBUG_TARGET_ASSIGNED_DIR = (
    GRASP_DEBUG_PREVIEW_DIR / "target_assigned"
)
GRASP_DEBUG_SELECTED_GRASP_DIR = (
    GRASP_DEBUG_PREVIEW_DIR / "selected_grasp"
)

LOG_OUTPUT_DIR = (
    CLOSED_LOOP_OUTPUT_DIR / "logs"
)

LEGACY_WORKSPACE_PREVIEW_DIR = (
    SCRIPT_DIR / "workspace_preview"
)
LEGACY_VIEW_TESTS_DIR = (
    SCRIPT_DIR / "view_tests"
)

GROUNDING_SCRIPT = SCRIPT_DIR / "run_groundingdino_inference.py"
POINTCLOUD_FUSION_SCRIPT = SCRIPT_DIR / "run_pointcloud_fusion.py"
LEGACY_PERCEPTION_PYTHON = os.environ.get(
    "LEGACY_PERCEPTION_PYTHON"
)

if not LEGACY_PERCEPTION_PYTHON:
    raise RuntimeError(
        "LEGACY_PERCEPTION_PYTHON is not set. "
        "Set it to the Python interpreter used for GroundingDINO "
        "and source-style point-cloud fusion."
    )

GROUNDING_PYTHON = Path(
    LEGACY_PERCEPTION_PYTHON
).expanduser().resolve()
GROUNDING_BOX_THRESHOLD = 0.15
GROUNDING_TEXT_THRESHOLD = 0.25
GROUNDING_CROP_MARGIN = 20

# VLM and GroundingDINO receive the direct top-view RGB frame.
PERCEPTION_WIDTH = 640
PERCEPTION_HEIGHT = 640

# Full-scene fallback uses the SAME official workspace stored in
# env.perception_workspace_limits. That workspace is derived once from the
# fixed purple crop with the Panda-facing X-min edge moved inward by 5 cm.
FULL_SCENE_TABLE_HEIGHT_M = 0.855
FULL_SCENE_TABLE_CLEARANCE_M = 0.003
FULL_SCENE_MAX_HEIGHT_ABOVE_TABLE_M = 0.30

FULL_SCENE_CAMERAS = (
    ("topview", 640, 640),
    ("frontview", 640, 480),
    ("left_oblique_25deg", 640, 480),
    ("right_oblique_25deg", 640, 480),
)

# A candidate bbox is retained only when enough of its valid 3D points
# remain inside the calibrated world-coordinate workspace.
MIN_WORKSPACE_POINTS = 30

MAX_ATTEMPTS = 50
MIN_GRASPED_GRIPPER_WIDTH = 0.005

# Original PyBullet ThinkGrasp grasp-assignment / filtering constants.
# These mirror utils.graspnet_config in the source implementation.
PYBULLET_GRASP_OBJECT_DISTANCE_M = 0.05
PYBULLET_GRASP_ANGLE_DEG = 15.0

# ---------------------------------------------------------------------------
# Grasp-control integration switch
# ---------------------------------------------------------------------------
#
# Keep the old OSC execution path available for immediate rollback.
#
# First formal IK integration stage intentionally stops after reaching the
# real GraspNet-derived pregrasp pose. It does NOT descend, close, lift,
# place, or enter the old OSC recovery path.
GRASP_CONTROL_MODE = "ik"  # "ik" or "osc"
IK_STOP_AFTER_PREGRASP = False
IK_STOP_AFTER_GRASP_POSE = False


# q_ref advances at MuJoCo's 500 Hz physics rate. Capture every 50 physics
# steps = one frame every 0.1 s = 10 real-time frames/s, matching the
# DualViewRecorder output fps below.
IK_VIDEO_PHYSICS_CAPTURE_INTERVAL = 50

# PyBullet-style fixed Panda drop posture.
PANDA_DROP_JOINTS = np.array(
    [
         0.86239812,
         0.95242016,
        -0.42055549,
        -1.04547415,
         0.36347610,
         1.93136290,
         2.83199626,
    ],
    dtype=np.float64,
)


def load_case_config():
    """Load one fixed MuJoCo testcase in the same spirit as PyBullet presets.

    File format:
        line 1: natural-language language goal
        line 2: fixed MuJoCo GT target object name

    The language goal is used only by VLM / GroundingDINO / grasp planning.
    The GT target object is independent of VLM output and is used only for
    simulator-side task-success evaluation after grasp execution.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Run one fixed ThinkGrasp MuJoCo testcase. "
            "The testcase stores both the language goal and the simulator "
            "ground-truth target object."
        )
    )
    parser.add_argument(
        "--case",
        type=str,
        default=str(DEFAULT_CASE_PATH),
        help=(
            "Case file path or case name under ./cases. "
            "Default: cases/case02_white_ramekin.txt"
        ),
    )
    args = parser.parse_args()

    requested = Path(args.case)

    if not requested.is_absolute():
        # A bare name such as "case01_red_mug" resolves under ./cases.
        if requested.parent == Path("."):
            if requested.suffix == "":
                requested = requested.with_suffix(".txt")
            requested = CASE_DIR / requested
        else:
            requested = SCRIPT_DIR / requested

    case_path = requested.resolve()

    if not case_path.is_file():
        raise FileNotFoundError(
            f"Case file does not exist: {case_path}"
        )

    lines = case_path.read_text(
        encoding="utf-8"
    ).splitlines()

    if len(lines) < 2:
        raise ValueError(
            "Case file must contain at least two lines: "
            "line 1 = language goal, "
            "line 2 = MuJoCo GT target object name."
        )

    language_goal = lines[0].strip()
    target_object = lines[1].strip()

    if not language_goal:
        raise ValueError(
            f"Case language goal is empty: {case_path}"
        )

    if not target_object:
        raise ValueError(
            f"Case GT target object is empty: {case_path}"
        )

    return {
        "case_path": case_path,
        "language_goal": language_goal,
        "target_object": target_object,
    }



def identify_highest_scene_object(env):
    """Return the MuJoCo scene object whose body center has the highest Z.

    This intentionally mirrors the original PyBullet ThinkGrasp heuristic:
    after a successful lift, the highest rigid object is treated as the object
    currently held by the gripper. This is simulator ground truth and is used
    only for task-success evaluation, not for perception or grasp generation.
    """

    object_heights = []

    for object_name, body_id in env.object_body_ids.items():
        body_id = int(body_id)
        body_position = np.asarray(
            env.sim.data.body_xpos[body_id],
            dtype=np.float64,
        ).copy()

        object_heights.append(
            {
                "name": str(object_name),
                "body_id": body_id,
                "position": body_position,
                "z": float(body_position[2]),
            }
        )

    if not object_heights:
        return {
            "name": None,
            "body_id": None,
            "position": None,
            "z": None,
            "all_objects": [],
        }

    highest = max(
        object_heights,
        key=lambda item: item["z"],
    )

    return {
        "name": highest["name"],
        "body_id": highest["body_id"],
        "position": highest["position"],
        "z": highest["z"],
        "all_objects": object_heights,
    }


def preferred_location_to_world_point(
    topview_data,
    bbox_xyxy,
    preferred_location,
):
    """Convert a 1..9 preferred DINO-bbox cell to one valid world XYZ point.

    Cell numbering follows ThinkGrasp:

        1 2 3
        4 5 6
        7 8 9

    Preferred-point policy:
      1. Try the preferred cell centre pixel.
      2. If that pixel has no valid world point (including [0, 0, 0]),
         use the nearest valid pixel inside the same preferred cell.
      3. If the whole preferred cell is invalid, use the nearest valid pixel
         inside the complete GroundingDINO bbox.

    This keeps the original preferred-location intent while preventing an
    invalid default [0, 0, 0] from participating in XY grasp selection.
    """

    preferred_location = int(preferred_location)

    if not 1 <= preferred_location <= 9:
        preferred_location = 5

    pointcloud = np.asarray(
        topview_data["pointcloud"],
        dtype=np.float64,
    )

    if pointcloud.ndim != 3 or pointcloud.shape[2] != 3:
        raise ValueError(
            "Expected topview pointcloud with shape (H, W, 3), "
            f"got {pointcloud.shape}."
        )

    height, width = pointcloud.shape[:2]

    x1, y1, x2, y2 = np.asarray(
        bbox_xyxy,
        dtype=np.float64,
    ).reshape(4)

    x1 = float(np.clip(x1, 0, width - 1))
    x2 = float(np.clip(x2, 0, width - 1))
    y1 = float(np.clip(y1, 0, height - 1))
    y2 = float(np.clip(y2, 0, height - 1))

    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"Invalid GroundingDINO bbox: {bbox_xyxy}"
        )

    row = (preferred_location - 1) // 3
    col = (preferred_location - 1) % 3

    cell_x1 = x1 + col * (x2 - x1) / 3.0
    cell_x2 = x1 + (col + 1) * (x2 - x1) / 3.0
    cell_y1 = y1 + row * (y2 - y1) / 3.0
    cell_y2 = y1 + (row + 1) * (y2 - y1) / 3.0

    requested_center_x = int(
        np.clip(
            round((cell_x1 + cell_x2) / 2.0),
            0,
            width - 1,
        )
    )
    requested_center_y = int(
        np.clip(
            round((cell_y1 + cell_y2) / 2.0),
            0,
            height - 1,
        )
    )

    def _valid_world_point(point):
        point = np.asarray(point, dtype=np.float64).reshape(3)
        return bool(
            np.isfinite(point).all()
            and np.linalg.norm(point) > 1e-9
        )

    requested_world_point = pointcloud[
        requested_center_y,
        requested_center_x,
    ].copy()

    if _valid_world_point(requested_world_point):
        return {
            "preferred_location": preferred_location,
            "cell_xyxy": np.array(
                [cell_x1, cell_y1, cell_x2, cell_y2],
                dtype=np.float64,
            ),
            "requested_center_pixel_xy": np.array(
                [requested_center_x, requested_center_y],
                dtype=np.int64,
            ),
            "center_pixel_xy": np.array(
                [requested_center_x, requested_center_y],
                dtype=np.int64,
            ),
            "world_point": requested_world_point,
            "fallback_mode": "cell_center",
        }

    def _nearest_valid_pixel(search_xyxy):
        sx1, sy1, sx2, sy2 = np.asarray(
            search_xyxy,
            dtype=np.float64,
        ).reshape(4)

        ix1 = int(np.clip(np.floor(sx1), 0, width - 1))
        ix2 = int(np.clip(np.ceil(sx2), ix1 + 1, width))
        iy1 = int(np.clip(np.floor(sy1), 0, height - 1))
        iy2 = int(np.clip(np.ceil(sy2), iy1 + 1, height))

        region = pointcloud[iy1:iy2, ix1:ix2]

        valid_mask = (
            np.isfinite(region).all(axis=2)
            & (np.linalg.norm(region, axis=2) > 1e-9)
        )

        valid_y, valid_x = np.nonzero(valid_mask)

        if len(valid_x) == 0:
            return None

        global_x = valid_x + ix1
        global_y = valid_y + iy1

        pixel_dist_sq = (
            (global_x - requested_center_x) ** 2
            + (global_y - requested_center_y) ** 2
        )
        nearest = int(np.argmin(pixel_dist_sq))

        px = int(global_x[nearest])
        py = int(global_y[nearest])

        return (
            np.array([px, py], dtype=np.int64),
            pointcloud[py, px].copy(),
        )

    cell_fallback = _nearest_valid_pixel(
        [cell_x1, cell_y1, cell_x2, cell_y2]
    )

    if cell_fallback is not None:
        selected_pixel, selected_world_point = cell_fallback
        fallback_mode = "nearest_valid_in_preferred_cell"
    else:
        bbox_fallback = _nearest_valid_pixel(
            [x1, y1, x2, y2]
        )

        if bbox_fallback is None:
            raise RuntimeError(
                "Neither the preferred cell nor the complete GroundingDINO "
                "bbox contains a valid top-view 3D world point."
            )

        selected_pixel, selected_world_point = bbox_fallback
        fallback_mode = "nearest_valid_in_dino_bbox"

    return {
        "preferred_location": preferred_location,
        "cell_xyxy": np.array(
            [cell_x1, cell_y1, cell_x2, cell_y2],
            dtype=np.float64,
        ),
        "requested_center_pixel_xy": np.array(
            [requested_center_x, requested_center_y],
            dtype=np.int64,
        ),
        "center_pixel_xy": selected_pixel,
        "world_point": selected_world_point,
        "fallback_mode": fallback_mode,
    }



def assign_grasps_to_objects_pybullet_style(
    env,
    grasps,
    scores,
    angles_deg,
    distance_threshold=PYBULLET_GRASP_OBJECT_DISTANCE_M,
):
    """Assign each grasp to its nearest simulator object within 5 cm.

    This restores the original PyBullet ThinkGrasp semantics:
      - 5 cm assignment identifies which simulator object a grasp belongs to.
      - Grasps assigned to ANY object are retained.
      - The fixed testcase GT target is NOT used to pre-filter grasp choices.

    The returned per-grasp object identity is simulator-side bookkeeping for
    post-transport task evaluation only.
    """

    grasps = np.asarray(grasps, dtype=np.float64).reshape(-1, 7)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    angles_deg = np.asarray(angles_deg, dtype=np.float64).reshape(-1)

    if not (
        len(grasps) == len(scores) == len(angles_deg)
    ):
        raise ValueError(
            "Grasp / score / angle counts do not match: "
            f"{len(grasps)}, {len(scores)}, {len(angles_deg)}"
        )

    object_names = []
    object_body_ids = []
    object_centres = []

    for object_name, body_id in env.object_body_ids.items():
        object_names.append(str(object_name))
        object_body_ids.append(int(body_id))
        object_centres.append(
            np.asarray(
                env.sim.data.body_xpos[int(body_id)],
                dtype=np.float64,
            ).copy()
        )

    if not object_centres:
        raise RuntimeError(
            "No simulator objects are available for PyBullet-style "
            "grasp assignment."
        )

    object_centres = np.asarray(
        object_centres,
        dtype=np.float64,
    ).reshape(-1, 3)
    object_body_ids = np.asarray(
        object_body_ids,
        dtype=np.int64,
    )

    assigned_object_names = np.full(
        len(grasps),
        "",
        dtype=object,
    )
    assigned_object_body_ids = np.full(
        len(grasps),
        -1,
        dtype=np.int64,
    )
    assigned_object_distances = np.full(
        len(grasps),
        np.inf,
        dtype=np.float64,
    )

    for grasp_index, grasp in enumerate(grasps):
        distances = np.linalg.norm(
            object_centres - grasp[:3][None, :],
            axis=1,
        )

        within_threshold = np.flatnonzero(
            distances < float(distance_threshold)
        )

        if len(within_threshold) == 0:
            continue

        nearest_local = int(
            within_threshold[
                np.argmin(distances[within_threshold])
            ]
        )

        assigned_object_names[grasp_index] = (
            object_names[nearest_local]
        )
        assigned_object_body_ids[grasp_index] = int(
            object_body_ids[nearest_local]
        )
        assigned_object_distances[grasp_index] = float(
            distances[nearest_local]
        )

    assigned_indices = np.flatnonzero(
        assigned_object_body_ids >= 0
    )

    return {
        "grasps": grasps[assigned_indices],
        "scores": scores[assigned_indices],
        "angles_deg": angles_deg[assigned_indices],
        "source_indices": assigned_indices.astype(np.int64),
        "grasp_object_names": assigned_object_names[assigned_indices],
        "grasp_object_body_ids": assigned_object_body_ids[assigned_indices],
        "grasp_object_distances": assigned_object_distances[assigned_indices],
        "all_assigned_object_names": assigned_object_names,
        "all_assigned_object_body_ids": assigned_object_body_ids,
        "all_assigned_object_distances": assigned_object_distances,
        "object_names": np.asarray(object_names, dtype=object),
        "object_body_ids": object_body_ids,
        "object_centres": object_centres,
        "distance_threshold": float(distance_threshold),
    }



def select_grasp_pybullet_style(
    grasps,
    scores,
    angles_deg,
    object_names,
    preferred_world_point,
    angle_threshold_deg=PYBULLET_GRASP_ANGLE_DEG,
):
    """Mirror the ACTUAL original PyBullet ThinkGrasp grasp-selection flow.

    The original implementation applies the 15-degree preference separately
    for each simulator object after 5 cm grasp-to-object assignment:

      1. Split assigned grasps by simulator object.
      2. For each object:
           - if it has one or more grasps with approach angle <= 15 deg,
             keep only those grasps;
           - otherwise keep ALL grasps assigned to that object.
      3. Merge the surviving grasps from all objects.
      4. Select the surviving grasp nearest in XY to the preferred world
         point derived from VLM Preferred Grasping Location + DINO bbox.

    The fixed testcase GT target is NOT used to pre-filter this pool.
    GraspNet score is diagnostic only and does not decide final selection.
    """

    grasps = np.asarray(
        grasps,
        dtype=np.float64,
    ).reshape(-1, 7)

    scores = np.asarray(
        scores,
        dtype=np.float64,
    ).reshape(-1)

    angles_deg = np.asarray(
        angles_deg,
        dtype=np.float64,
    ).reshape(-1)

    object_names = np.asarray(
        object_names,
        dtype=object,
    ).reshape(-1)

    preferred_world_point = np.asarray(
        preferred_world_point,
        dtype=np.float64,
    ).reshape(3)

    if len(grasps) == 0:
        raise ValueError(
            "Cannot select from an empty grasp set."
        )

    if not (
        len(grasps)
        == len(scores)
        == len(angles_deg)
        == len(object_names)
    ):
        raise ValueError(
            "Grasp / score / angle / object-name counts do not match: "
            f"{len(grasps)}, {len(scores)}, "
            f"{len(angles_deg)}, {len(object_names)}"
        )

    xy_distances = np.linalg.norm(
        grasps[:, :2]
        - preferred_world_point[None, :2],
        axis=1,
    )

    # Preserve first-appearance object order for deterministic diagnostics.
    unique_object_names = []
    for object_name in object_names:
        object_name = str(object_name)
        if object_name not in unique_object_names:
            unique_object_names.append(object_name)

    surviving_indices = []
    per_object_angle_filter = []

    for object_name in unique_object_names:
        object_indices = np.flatnonzero(
            np.asarray(
                [
                    str(name) == object_name
                    for name in object_names
                ],
                dtype=bool,
            )
        )

        safe_indices = object_indices[
            angles_deg[object_indices]
            <= float(angle_threshold_deg)
        ]

        if len(safe_indices) > 0:
            kept_indices = safe_indices
            mode = "safe_cone_only"
        else:
            # Match the historical PyBullet implementation itself:
            # when this object has no <=15-degree grasp, retain all
            # grasps belonging to this object.
            kept_indices = object_indices
            mode = "all_object_grasps"

        surviving_indices.extend(
            int(index)
            for index in kept_indices
        )

        per_object_angle_filter.append(
            {
                "object_name": object_name,
                "assigned_indices": object_indices.copy(),
                "safe_indices": safe_indices.copy(),
                "kept_indices": kept_indices.copy(),
                "mode": mode,
            }
        )

    selection_pool = np.asarray(
        surviving_indices,
        dtype=np.int64,
    )

    if len(selection_pool) == 0:
        raise RuntimeError(
            "PyBullet-style object-wise angle filtering "
            "produced an empty selection pool."
        )

    selected_index = int(
        selection_pool[
            np.argmin(
                xy_distances[selection_pool]
            )
        ]
    )

    return {
        "selected_grasp": grasps[selected_index].copy(),
        "selected_index": selected_index,
        "selected_score": float(
            scores[selected_index]
        ),
        "selected_xy_distance": float(
            xy_distances[selected_index]
        ),
        "selected_angle_deg": float(
            angles_deg[selected_index]
        ),
        "angle_threshold_deg": float(
            angle_threshold_deg
        ),
        "per_object_angle_filter": (
            per_object_angle_filter
        ),
        "selection_pool_indices": (
            selection_pool.copy()
        ),
        "selection_pool_xy_distances": (
            xy_distances[selection_pool].copy()
        ),
        "selection_pool_scores": (
            scores[selection_pool].copy()
        ),
        "selection_pool_angles_deg": (
            angles_deg[selection_pool].copy()
        ),
        "selection_pool_object_names": (
            object_names[selection_pool].copy()
        ),
    }


def filter_grounding_candidates_to_perception_workspace(
    boxes,
    scores,
    phrases,
    crop_xyxy=FIXED_PERCEPTION_CROP_XYXY,
):
    """Keep GroundingDINO detections whose bbox centre is inside the purple crop.

    GroundingDINO still runs on the full 640x640 top-view image, so all bbox
    coordinates remain in the original image coordinate system. This avoids
    any crop-coordinate offset in the later 3D / preferred-cell pipeline.
    """

    boxes = np.asarray(
        boxes,
        dtype=np.float64,
    ).reshape(-1, 4)
    scores = np.asarray(
        scores,
        dtype=np.float64,
    ).reshape(-1)
    phrases = np.asarray(phrases)

    if len(boxes) == 0:
        return {
            "boxes": boxes,
            "scores": scores,
            "phrases": phrases,
            "keep_mask": np.zeros(0, dtype=bool),
            "centres_xy": np.zeros((0, 2), dtype=np.float64),
        }

    crop = np.asarray(
        crop_xyxy,
        dtype=np.float64,
    ).reshape(4)
    x1, y1, x2, y2 = crop

    centres_xy = np.column_stack(
        [
            0.5 * (boxes[:, 0] + boxes[:, 2]),
            0.5 * (boxes[:, 1] + boxes[:, 3]),
        ]
    )

    keep_mask = (
        (centres_xy[:, 0] >= x1)
        & (centres_xy[:, 0] <= x2)
        & (centres_xy[:, 1] >= y1)
        & (centres_xy[:, 1] <= y2)
    )

    return {
        "boxes": boxes[keep_mask],
        "scores": scores[keep_mask],
        "phrases": phrases[keep_mask],
        "keep_mask": keep_mask,
        "centres_xy": centres_xy,
    }



def run_source_style_pointcloud_fusion(
    raw_views_path,
    output_path,
):
    """Run original ThinkGrasp utils.process_pcds() in the legacy env.

    The MuJoCo process supplies four already world-frame, workspace-filtered
    camera clouds. The legacy worker then calls the ORIGINAL PyBullet
    ThinkGrasp `utils.process_pcds()` and `reconstruction_config` unchanged:
        statistical outlier removal
        -> normal estimation
        -> 1.5 mm voxel downsampling
        -> point-to-plane ICP
        -> merge
        -> voxel downsampling / normal re-estimation
    """

    command = [
        str(GROUNDING_PYTHON),
        str(POINTCLOUD_FUSION_SCRIPT),
        "--input",
        str(raw_views_path),
        "--output",
        str(output_path),
    ]

    print("Running source-style ThinkGrasp point-cloud fusion subprocess:")
    print(" ".join(command))

    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
    )

    if not Path(output_path).is_file():
        raise RuntimeError(
            "Source-style point-cloud fusion finished without creating "
            f"the expected output: {output_path}"
        )


def _derive_full_scene_workspace_xy_bounds(env):
    """Return XY bounds from the one official runtime MuJoCo workspace."""

    workspace_limits = np.asarray(
        env.perception_workspace_limits,
        dtype=np.float64,
    )

    if workspace_limits.shape != (3, 2):
        raise ValueError(
            "env.perception_workspace_limits must have shape (3, 2), "
            f"got {workspace_limits.shape}."
        )

    xy_min = workspace_limits[:2, 0].copy()
    xy_max = workspace_limits[:2, 1].copy()

    print(
        "Full-scene fallback uses official workspace XY min:",
        xy_min,
    )
    print(
        "Full-scene fallback uses official workspace XY max:",
        xy_max,
    )

    return xy_min, xy_max


def export_workspace_filtered_full_scene(
    env,
    output_path,
    ply_path,
):
    """Export four-view full scene using original ThinkGrasp fusion logic.

    MuJoCo-specific part:
        capture the four current MuJoCo cameras
        -> use their already world-frame point clouds
        -> keep the validated official MuJoCo workspace
        -> keep the existing tabletop clearance

    Source-faithful part:
        each camera cloud is sorted by Z as in get_fuse_pointcloud()
        -> legacy subprocess calls local source copy process_pcds()
           with the copied original reconstruction_config unchanged.
    """

    workspace_xy_min, workspace_xy_max = (
        _derive_full_scene_workspace_xy_bounds(
            env=env,
        )
    )

    official_workspace = np.asarray(
        env.perception_workspace_limits,
        dtype=np.float64,
    )

    # Preserve the already-validated MuJoCo geometric workspace policy.
    # The point-cloud FUSION algorithm itself is delegated unchanged to the
    # original ThinkGrasp source implementation.
    z_min = max(
        float(official_workspace[2, 0]),
        float(FULL_SCENE_TABLE_HEIGHT_M)
        + float(FULL_SCENE_TABLE_CLEARANCE_M),
    )
    z_max = min(
        float(official_workspace[2, 1]),
        float(FULL_SCENE_TABLE_HEIGHT_M)
        + float(FULL_SCENE_MAX_HEIGHT_ABOVE_TABLE_M),
    )

    if z_min >= z_max:
        raise RuntimeError(
            "Official workspace Z range is empty after tabletop clearance."
        )

    raw_view_payload = {}
    per_camera_raw_counts = {}
    per_camera_kept_counts = {}

    for camera_index, (
        camera_name,
        width,
        height,
    ) in enumerate(FULL_SCENE_CAMERAS):
        camera_data = env.get_camera_data(
            camera_name=camera_name,
            width=int(width),
            height=int(height),
        )

        points_image = np.asarray(
            camera_data["pointcloud"],
            dtype=np.float32,
        )
        depth = np.asarray(
            camera_data["depth"],
            dtype=np.float32,
        )
        colors_image = np.asarray(
            camera_data["color"]
        )

        base_valid = (
            np.isfinite(points_image).all(axis=-1)
            & np.isfinite(depth)
            & (depth > 0.0)
            & (depth < 3.0)
        )

        per_camera_raw_counts[camera_name] = int(
            np.count_nonzero(base_valid)
        )

        # Match original get_fuse_pointcloud() bound convention:
        # lower bound inclusive, upper bound exclusive.
        workspace_mask = (
            base_valid
            & (
                points_image[..., 0]
                >= float(workspace_xy_min[0])
            )
            & (
                points_image[..., 0]
                < float(workspace_xy_max[0])
            )
            & (
                points_image[..., 1]
                >= float(workspace_xy_min[1])
            )
            & (
                points_image[..., 1]
                < float(workspace_xy_max[1])
            )
            & (
                points_image[..., 2]
                >= float(z_min)
            )
            & (
                points_image[..., 2]
                < float(z_max)
            )
        )

        kept_points = points_image[workspace_mask]
        kept_colors = colors_image[workspace_mask]

        if len(kept_points) == 0:
            raise RuntimeError(
                "Source-style full-scene fusion received an empty "
                f"workspace cloud from camera {camera_name!r}."
            )

        # Original get_fuse_pointcloud() sorts each camera cloud by Z before
        # creating the Open3D point cloud.
        z_order = np.argsort(
            kept_points[:, 2]
        )
        kept_points = kept_points[z_order]
        kept_colors = kept_colors[z_order]

        if np.issubdtype(
            kept_colors.dtype,
            np.integer,
        ):
            kept_colors = (
                kept_colors.astype(np.float32)
                / 255.0
            )
        else:
            kept_colors = kept_colors.astype(
                np.float32
            )
            if (
                len(kept_colors) > 0
                and np.max(kept_colors) > 1.0
            ):
                kept_colors /= 255.0

        raw_view_payload[
            f"points_{camera_index}"
        ] = kept_points.astype(np.float32)
        raw_view_payload[
            f"colors_{camera_index}"
        ] = kept_colors.astype(np.float32)

        per_camera_kept_counts[camera_name] = int(
            len(kept_points)
        )

    raw_view_payload["camera_names"] = np.asarray(
        [
            camera_name
            for camera_name, _, _
            in FULL_SCENE_CAMERAS
        ]
    )
    raw_view_payload["camera_count"] = np.asarray(
        len(FULL_SCENE_CAMERAS),
        dtype=np.int64,
    )
    raw_view_payload["workspace_xy_min"] = (
        workspace_xy_min.astype(np.float64)
    )
    raw_view_payload["workspace_xy_max"] = (
        workspace_xy_max.astype(np.float64)
    )
    raw_view_payload["workspace_z_min"] = np.asarray(
        z_min,
        dtype=np.float64,
    )
    raw_view_payload["workspace_z_max"] = np.asarray(
        z_max,
        dtype=np.float64,
    )

    FULL_SCENE_RAW_VIEWS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    np.savez_compressed(
        FULL_SCENE_RAW_VIEWS_PATH,
        **raw_view_payload,
    )

    output_path = Path(output_path)
    ply_path = Path(ply_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    ply_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_source_style_pointcloud_fusion(
        raw_views_path=FULL_SCENE_RAW_VIEWS_PATH,
        output_path=output_path,
    )

    with np.load(
        output_path,
        allow_pickle=False,
    ) as fused_data:
        points = fused_data[
            "points"
        ].astype(np.float32)
        colors = fused_data[
            "colors"
        ].astype(np.float32)

        source_voxel_size = float(
            fused_data["voxel_size"]
        )
        source_nb_neighbors = int(
            fused_data["nb_neighbors"]
        )
        source_std_ratio = float(
            fused_data["std_ratio"]
        )
        source_max_correspondence_distance = float(
            fused_data["max_correspondence_distance"]
        )

    export_colored_pointcloud_ply(
        output_path=ply_path,
        points=points,
        colors=colors,
    )

    print()
    print(
        "Source-style full-scene fallback exported."
    )
    print(
        "Full-scene workspace world XY min:",
        workspace_xy_min,
    )
    print(
        "Full-scene workspace world XY max:",
        workspace_xy_max,
    )
    print(
        "Full-scene workspace world Z:",
        [z_min, z_max],
    )
    print(
        "Full-scene per-camera raw points:",
        per_camera_raw_counts,
    )
    print(
        "Full-scene per-camera kept points:",
        per_camera_kept_counts,
    )
    print(
        "Original ThinkGrasp fusion config:",
        {
            "nb_neighbors": source_nb_neighbors,
            "std_ratio": source_std_ratio,
            "voxel_size": source_voxel_size,
            "max_correspondence_distance": (
                source_max_correspondence_distance
            ),
        },
    )
    print(
        "Full-scene source-fused points:",
        len(points),
    )
    print(
        "Full-scene filtered XYZ min:",
        points.min(axis=0),
    )
    print(
        "Full-scene filtered XYZ max:",
        points.max(axis=0),
    )
    print(
        "Full-scene GraspNet input PLY saved:",
        ply_path,
    )

    return {
        "point_count": int(len(points)),
        "workspace_xy_min": (
            workspace_xy_min.copy()
        ),
        "workspace_xy_max": (
            workspace_xy_max.copy()
        ),
        "workspace_z_min": float(z_min),
        "workspace_z_max": float(z_max),
        "camera_raw_counts": (
            per_camera_raw_counts
        ),
        "camera_kept_counts": (
            per_camera_kept_counts
        ),
        "source_voxel_size": source_voxel_size,
    }


def filter_full_scene_grasps_to_target_xy(
    grasps,
    scores,
    angles_deg,
    target_xy_min,
    target_xy_max,
):
    """Keep fallback grasps whose centres lie inside the target XY crop."""

    grasps = np.asarray(
        grasps,
        dtype=np.float64,
    ).reshape(-1, 7)
    scores = np.asarray(
        scores,
        dtype=np.float64,
    ).reshape(-1)
    angles_deg = np.asarray(
        angles_deg,
        dtype=np.float64,
    ).reshape(-1)

    target_xy_min = np.asarray(
        target_xy_min,
        dtype=np.float64,
    ).reshape(2)
    target_xy_max = np.asarray(
        target_xy_max,
        dtype=np.float64,
    ).reshape(2)

    if len(grasps) == 0:
        keep_indices = np.empty(
            (0,),
            dtype=np.int64,
        )
    else:
        centres_xy = grasps[:, :2]
        keep_mask = (
            (
                centres_xy[:, 0]
                >= target_xy_min[0]
            )
            & (
                centres_xy[:, 0]
                <= target_xy_max[0]
            )
            & (
                centres_xy[:, 1]
                >= target_xy_min[1]
            )
            & (
                centres_xy[:, 1]
                <= target_xy_max[1]
            )
        )
        keep_indices = np.flatnonzero(
            keep_mask
        )

    return {
        "grasps": grasps[keep_indices],
        "scores": scores[keep_indices],
        "angles_deg": angles_deg[keep_indices],
        "source_indices": keep_indices,
    }


def run_groundingdino_detection(
    image,
    text_prompt,
):
    """Run GroundingDINO in the legacy ThinkGrasp environment."""

    BRIDGE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    imageio.imwrite(
        GROUNDING_IMAGE_PATH,
        np.asarray(image),
    )

    command = [
        str(GROUNDING_PYTHON),
        str(GROUNDING_SCRIPT),
        "--image",
        str(GROUNDING_IMAGE_PATH),
        "--prompt",
        text_prompt,
        "--output",
        str(GROUNDING_RESULT_PATH),
        "--visualization",
        str(GROUNDING_VIS_PATH),
        "--box-threshold",
        str(GROUNDING_BOX_THRESHOLD),
        "--text-threshold",
        str(GROUNDING_TEXT_THRESHOLD),
        "--device",
        "cuda",
    ]

    print("Running GroundingDINO subprocess:")
    print(" ".join(command))

    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
    )

    result = np.load(
        GROUNDING_RESULT_PATH,
        allow_pickle=True,
    )

    boxes = np.asarray(
        result["boxes_xyxy"],
        dtype=np.float64,
    )
    scores = np.asarray(
        result["scores"],
        dtype=np.float64,
    )
    phrases = np.asarray(
        result["phrases"],
    )

    if len(boxes) == 0:
        return {
            "boxes": boxes,
            "scores": scores,
            "phrases": phrases,
            "best_box": None,
            "best_score": None,
            "best_phrase": None,
        }

    best_index = int(np.argmax(scores))

    return {
        "boxes": boxes,
        "scores": scores,
        "phrases": phrases,
        "best_box": boxes[best_index],
        "best_score": float(scores[best_index]),
        "best_phrase": str(phrases[best_index]),
    }









def _build_grasp_debug_marker_points(
    env,
    grasp,
):
    """Return world-frame marker points for one ThinkGrasp 7D grasp.

    The visual marker matches the existing selected-grasp diagnostic:
        - two parallel-jaw fingers
        - rear bridge
        - approach-direction segment
        - grasp centre
    """

    grasp = np.asarray(
        grasp,
        dtype=np.float64,
    ).reshape(7)

    grasp_center = grasp[:3]

    grasp_eef_pose = (
        env.grasp_tip_pose_to_eef_pose(
            grasp
        )
    )
    rotation = grasp_eef_pose[:3, :3]

    # Current ThinkGrasp convention used throughout this runner:
    # local +Z is the approaching direction.
    opening_axis = rotation[:, 0]
    approach_axis = rotation[:, 2]

    half_opening = 0.035
    finger_length = 0.060
    approach_line_length = 0.090

    left_front = (
        grasp_center
        + half_opening * opening_axis
    )
    right_front = (
        grasp_center
        - half_opening * opening_axis
    )

    left_rear = (
        left_front
        - finger_length * approach_axis
    )
    right_rear = (
        right_front
        - finger_length * approach_axis
    )

    approach_start = (
        grasp_center
        - approach_line_length * approach_axis
    )
    approach_end = grasp_center

    return np.stack(
        [
            grasp_center,
            left_front,
            right_front,
            left_rear,
            right_rear,
            approach_start,
            approach_end,
        ],
        axis=0,
    )









































def _sample_grasp_debug_line_3d(
    start,
    end,
    spacing=0.0015,
    thickness=0.0015,
):
    """Sample one visible 3D line as a small thick point cloud."""

    start = np.asarray(
        start,
        dtype=np.float64,
    ).reshape(3)

    end = np.asarray(
        end,
        dtype=np.float64,
    ).reshape(3)

    length = float(
        np.linalg.norm(end - start)
    )

    sample_count = max(
        2,
        int(np.ceil(length / float(spacing))) + 1,
    )

    line_points = np.linspace(
        start,
        end,
        sample_count,
    )

    r = float(thickness)

    offsets = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [ r, 0.0, 0.0],
            [-r, 0.0, 0.0],
            [0.0,  r, 0.0],
            [0.0, -r, 0.0],
            [0.0, 0.0,  r],
            [0.0, 0.0, -r],
        ],
        dtype=np.float64,
    )

    return (
        line_points[:, None, :]
        + offsets[None, :, :]
    ).reshape(-1, 3)







def _build_grasp_debug_pointcloud(
    env,
    grasps,
):
    """Convert one or more grasps into 3D marker points."""

    grasps = np.asarray(
        grasps,
        dtype=np.float64,
    )

    if grasps.size == 0:
        return (
            np.empty(
                (0, 3),
                dtype=np.float32,
            ),
            np.empty(
                (0, 3),
                dtype=np.float32,
            ),
        )

    grasps = grasps.reshape(-1, 7)

    all_points = []
    all_colors = []

    for grasp in grasps:
        marker = _build_grasp_debug_marker_points(
            env=env,
            grasp=grasp,
        )

        (
            center,
            left_front,
            right_front,
            left_rear,
            right_rear,
            approach_start,
            approach_end,
        ) = marker

        finger_points = np.concatenate(
            [
                _sample_grasp_debug_line_3d(
                    left_rear,
                    left_front,
                ),
                _sample_grasp_debug_line_3d(
                    right_rear,
                    right_front,
                ),
                _sample_grasp_debug_line_3d(
                    left_rear,
                    right_rear,
                ),
            ],
            axis=0,
        )

        finger_colors = np.tile(
            np.asarray(
                [[1.0, 0.0, 1.0]],
                dtype=np.float32,
            ),
            (
                len(finger_points),
                1,
            ),
        )

        approach_points = (
            _sample_grasp_debug_line_3d(
                approach_start,
                approach_end,
            )
        )

        approach_colors = np.tile(
            np.asarray(
                [[1.0, 0.235, 0.0]],
                dtype=np.float32,
            ),
            (
                len(approach_points),
                1,
            ),
        )

        center_radius = 0.003

        center_offsets = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [ center_radius, 0.0, 0.0],
                [-center_radius, 0.0, 0.0],
                [0.0,  center_radius, 0.0],
                [0.0, -center_radius, 0.0],
                [0.0, 0.0,  center_radius],
                [0.0, 0.0, -center_radius],
            ],
            dtype=np.float64,
        )

        center_points = (
            center[None, :]
            + center_offsets
        )

        center_colors = np.ones(
            (
                len(center_points),
                3,
            ),
            dtype=np.float32,
        )

        all_points.extend(
            [
                finger_points.astype(
                    np.float32
                ),
                approach_points.astype(
                    np.float32
                ),
                center_points.astype(
                    np.float32
                ),
            ]
        )

        all_colors.extend(
            [
                finger_colors,
                approach_colors,
                center_colors,
            ]
        )

    return (
        np.concatenate(
            all_points,
            axis=0,
        ),
        np.concatenate(
            all_colors,
            axis=0,
        ),
    )


def prepare_full_scene_grasp_debug_background(
    env,
):
    """Build the full-scene cloud once per attempt."""

    temporary_ply_path = (
        BRIDGE_DIR
        / "grasp_debug_full_scene_background.ply"
    )

    export_workspace_filtered_full_scene(
        env=env,
        output_path=FULL_SCENE_PATH,
        ply_path=temporary_ply_path,
    )

    with np.load(
        FULL_SCENE_PATH,
        allow_pickle=False,
    ) as fused_data:
        scene_points = np.asarray(
            fused_data["points"],
            dtype=np.float32,
        ).reshape(-1, 3)

        scene_colors = np.asarray(
            fused_data["colors"],
            dtype=np.float32,
        ).reshape(-1, 3)

    if (
        len(scene_colors) > 0
        and np.max(scene_colors) > 1.0
    ):
        scene_colors = (
            scene_colors / 255.0
        )

    scene_colors = np.clip(
        scene_colors,
        0.0,
        1.0,
    )

    try:
        temporary_ply_path.unlink()
    except FileNotFoundError:
        pass

    return (
        scene_points,
        scene_colors,
    )


def save_grasp_set_debug_ply(
    env,
    grasps,
    scene_points,
    scene_colors,
    output_dir,
    output_name,
):
    """Save one grasp set over the shared full-scene cloud."""

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    grasp_points, grasp_colors = (
        _build_grasp_debug_pointcloud(
            env=env,
            grasps=grasps,
        )
    )

    output_points = np.concatenate(
        [
            np.asarray(
                scene_points,
                dtype=np.float32,
            ).reshape(-1, 3),
            grasp_points,
        ],
        axis=0,
    )

    output_colors = np.concatenate(
        [
            np.asarray(
                scene_colors,
                dtype=np.float32,
            ).reshape(-1, 3),
            grasp_colors,
        ],
        axis=0,
    )

    output_path = (
        output_dir
        / str(output_name)
    )

    exported_path = (
        export_colored_pointcloud_ply(
            output_path=output_path,
            points=output_points,
            colors=output_colors,
        )
    )

    grasp_array = np.asarray(
        grasps,
        dtype=np.float64,
    )

    grasp_count = (
        len(
            grasp_array.reshape(-1, 7)
        )
        if grasp_array.size
        else 0
    )

    print(
        "Grasp-debug PLY:",
        exported_path,
    )
    print(
        "  full-scene points:",
        len(scene_points),
    )
    print(
        "  grasp count:",
        grasp_count,
    )
    print(
        "  grasp-marker points:",
        len(grasp_points),
    )

    return exported_path


def clear_previous_closed_loop_outputs():
    """Remove outputs from the previous closed-loop run.

    Historical logs and videos are preserved.
    """

    cleanup_directories = (
        BRIDGE_DIR,
        VLM_SELECTION_OUTPUT_DIR,
        GROUNDING_GRID_OUTPUT_DIR,
        FUSED_CLOUD_PREVIEW_DIR,
        PERCEPTION_VIEW_OUTPUT_DIR,
        GRASP_DEBUG_PREVIEW_DIR,
        LEGACY_WORKSPACE_PREVIEW_DIR,
        LEGACY_VIEW_TESTS_DIR,
    )

    removable_suffixes = {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".webp",
        ".ply",
        ".npz",
    }

    removed_paths = []

    for directory in cleanup_directories:
        if not directory.exists():
            continue

        for path in directory.rglob("*"):
            if (
                path.is_file()
                and path.suffix.lower() in removable_suffixes
            ):
                path.unlink()
                removed_paths.append(path)

    print()
    print(
        "Previous closed-loop output files removed:",
        len(removed_paths),
    )

    for path in removed_paths:
        print("Removed output:", path)



class _TeeStream:
    """Write the same text to the terminal and one log file."""

    def __init__(
        self,
        terminal_stream,
        log_file,
    ):
        self.terminal_stream = terminal_stream
        self.log_file = log_file

    def write(self, text):
        self.terminal_stream.write(text)
        self.log_file.write(text)
        return len(text)

    def flush(self):
        self.terminal_stream.flush()
        self.log_file.flush()

    def isatty(self):
        return self.terminal_stream.isatty()

    @property
    def encoding(self):
        return getattr(
            self.terminal_stream,
            "encoding",
            "utf-8",
        )


def _run_with_log():
    """Run one closed-loop task while teeing stdout/stderr to a log."""

    LOG_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )
    log_path = (
        LOG_OUTPUT_DIR
        / f"{timestamp}.log"
    )

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    with open(
        log_path,
        "w",
        encoding="utf-8",
        buffering=1,
    ) as log_file:
        sys.stdout = _TeeStream(
            original_stdout,
            log_file,
        )
        sys.stderr = _TeeStream(
            original_stderr,
            log_file,
        )

        try:
            print(
                "Closed-loop log:",
                log_path,
            )
            main()
        except Exception:
            traceback.print_exc()
            raise SystemExit(1)
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def summarize_ik_phase_diagnostic(phase_result):
    """Aggregate existing q_ref diagnostics for one IK motion phase.

    Diagnostic only. Handles both:
      - move_pose_ik() result used by pregrasp;
      - straight_move_ik() result containing multiple IK waypoints.
    """

    if phase_result is None:
        return None

    motion_results = []

    direct_motion = phase_result.get("motion")
    if isinstance(direct_motion, dict):
        motion_results.append(direct_motion)

    for waypoint_result in phase_result.get("waypoints", []) or []:
        if not isinstance(waypoint_result, dict):
            continue
        waypoint_motion = waypoint_result.get("motion")
        if isinstance(waypoint_motion, dict):
            motion_results.append(waypoint_motion)

    if not motion_results:
        return {
            "motion_segments": 0,
            "physics_steps": 0,
            "simulated_seconds": 0.0,
            "max_tracking_error_seen": None,
            "max_raw_torque_seen": None,
            "max_relevant_contact_count": 0,
            "diagnostic_contact_pairs": [],
            "max_eef_force_metric_seen": None,
            "force_stop_triggered": False,
            "force_stop_metric": None,
            "force_stop_wrench": None,
        }

    tracking_values = [
        float(item["max_tracking_error_seen"])
        for item in motion_results
        if item.get("max_tracking_error_seen") is not None
    ]
    torque_values = [
        float(item["max_raw_torque_seen"])
        for item in motion_results
        if item.get("max_raw_torque_seen") is not None
    ]

    force_metric_values = [
        float(item["max_eef_force_metric_seen"])
        for item in motion_results
        if item.get("max_eef_force_metric_seen") is not None
    ]
    force_stop_items = [
        item for item in motion_results
        if item.get("stopped_by_force", False)
    ]

    contact_pairs = set()
    for item in motion_results:
        contact_pairs.update(
            str(pair)
            for pair in item.get(
                "diagnostic_contact_pairs",
                [],
            )
        )

    return {
        "motion_segments": len(motion_results),
        "physics_steps": int(
            sum(
                int(item.get("physics_steps", 0))
                for item in motion_results
            )
        ),
        "simulated_seconds": float(
            sum(
                float(item.get("simulated_seconds", 0.0))
                for item in motion_results
            )
        ),
        "max_tracking_error_seen": (
            max(tracking_values)
            if tracking_values
            else None
        ),
        "max_raw_torque_seen": (
            max(torque_values)
            if torque_values
            else None
        ),
        "max_relevant_contact_count": int(
            max(
                (
                    int(item.get("max_relevant_contact_count", 0))
                    for item in motion_results
                ),
                default=0,
            )
        ),
        "diagnostic_contact_pairs": sorted(contact_pairs),
        "max_eef_force_metric_seen": (
            max(force_metric_values) if force_metric_values else None
        ),
        "force_stop_triggered": bool(force_stop_items),
        "force_stop_metric": (
            force_stop_items[-1].get("force_stop_metric")
            if force_stop_items else None
        ),
        "force_stop_wrench": (
            force_stop_items[-1].get("force_stop_wrench")
            if force_stop_items else None
        ),
    }



def main():
    case_config = load_case_config()
    vlm_goal = case_config["language_goal"]
    target_object = case_config["target_object"]

    clear_previous_closed_loop_outputs()

    print("Case file:", case_config["case_path"])
    print("Fixed language goal:", vlm_goal)
    print("Simulator GT target for final task evaluation only:", target_object)

    if GRASP_CONTROL_MODE == "ik":
        controller_config = (
            load_thinkgrasp_joint_position_controller_config()
        )
    elif GRASP_CONTROL_MODE == "osc":
        controller_config = load_composite_controller_config(
            controller="BASIC"
        )
    else:
        raise ValueError(
            "GRASP_CONTROL_MODE must be 'ik' or 'osc', got "
            f"{GRASP_CONTROL_MODE!r}"
        )

    print("Grasp control mode:", GRASP_CONTROL_MODE)

    if GRASP_CONTROL_MODE == "ik":
        print(
            "IK staged integration:",
            {
                "stop_after_pregrasp": IK_STOP_AFTER_PREGRASP,
                "stop_after_grasp_pose": IK_STOP_AFTER_GRASP_POSE,
            },
        )

    env = ThinkGraspMinimalEnv(
        controller_configs=controller_config,
        hard_reset=False,
    )

    if target_object not in env.object_body_ids:
        raise ValueError(
            "Case GT target object "
            f"{target_object!r} is not present in the MuJoCo scene. "
            f"Available objects: {sorted(env.object_body_ids.keys())}"
        )

    target_body_id = int(
        env.object_body_ids[target_object]
    )

    env.set_language_task(
        goal=vlm_goal,
        target_object_name=target_object,
    )

    print(
        "Simulator GT target body id:",
        target_body_id,
    )

    recorder = DualViewRecorder(
        env,
        output_dir="grasp_videos",
        camera_left="frontview",
        camera_right="sideview",
        capture_interval=2,
        fps=10,
    )

    try:
        # ThinkGraspMinimalEnv construction already performs the initial reset
        # and clutter generation. Do not reset again here, otherwise a second
        # valid clutter scene would be generated and replace the first one.

        # 保存初始化后的安全位姿和7个Panda关节角。
        # 失败恢复时，只有EEF pose和关节构型都回到home附近，
        # 才允许重新感知。
        home_eef_pose = env.get_eef_pose().copy()
        home_joint_positions = (
            env.get_arm_joint_positions().copy()
        )

        print(
            "Saved home joint positions:",
            home_joint_positions,
        )
        print(
            "Official MuJoCo fixed workspace:",
            env.perception_workspace_limits,
        )

        recorder.start()
        recorder.add_hold(1.0)

        env.open_gripper(steps=30)

        target_completed = False
        diagnostic_completed = False
        diagnostic_stop_stage = None
        ik_failure_stopped_safely = False
        ik_failure_phase = None
        attempts_started = 0
        ended_no_grasp_after_fallback = False

        print()
        print("#" * 60)
        print("Language goal:", vlm_goal)
        print("#" * 60)
        print("Maximum attempts:", MAX_ATTEMPTS)

        for attempt in range(1, MAX_ATTEMPTS + 1):
            attempts_started = attempt

            print()
            print("=" * 60)
            print(f"Attempt {attempt}/{MAX_ATTEMPTS}")
            print("=" * 60)

            recorder.capture_frame()
            recorder.add_hold(0.5)

            # =====================================================
            # Source-faithful PyBullet perception path:
            #   raw top-view RGB-D -> orthographic workspace heightmap
            #   same heightmap RGB -> VLM / GroundingDINO
            #   GroundingDINO bbox -> PyBullet-style workspace-grid crop
            # =====================================================
            perception_camera_name = "topview"

            raw_topview_data = env.get_camera_data(
                camera_name=perception_camera_name,
                width=PERCEPTION_WIDTH,
                height=PERCEPTION_HEIGHT,
            )
            raw_topview_data["camera_name"] = (
                perception_camera_name
            )

            topview_data = build_pybullet_style_heightmap(
                topview_data=raw_topview_data,
                workspace_limits=(
                    env.perception_workspace_limits
                ),
                pixel_size=0.002,
            )

            BRIDGE_DIR.mkdir(
                parents=True,
                exist_ok=True,
            )
            imageio.imwrite(
                FULL_PERCEPTION_IMAGE_PATH,
                np.asarray(topview_data["color"]),
            )

            print(
                "GroundingDINO perception representation:",
                "PyBullet-style orthographic workspace heightmap",
            )
            print(
                "3D perception workspace:",
                env.perception_workspace_limits,
            )
            print(
                "Raw top-view RGB shape:",
                raw_topview_data["color"].shape,
            )
            print(
                "PyBullet-style heightmap RGB shape:",
                topview_data["color"].shape,
            )
            print(
                "PyBullet-style heightmap valid workspace points:",
                topview_data["valid_workspace_point_count"],
            )

            # =====================================================
            # VLM selection:
            #   external natural-language goal + same top-view RGB
            #   -> selected object + preferred 3x3 grasp location
            # =====================================================

            print()
            print("Running Qwen3-VL selection.")
            print("VLM goal:", vlm_goal)

            vlm_result = run_vlm_selection(
                image_path=FULL_PERCEPTION_IMAGE_PATH,
                goal=vlm_goal,
                system_prompt_path=VLM_SYSTEM_PROMPT_PATH,
            )

            vlm_selected_object = str(
                vlm_result["selected_object"]
            ).strip()

            preferred_grasping_location = int(
                vlm_result[
                    "preferred_grasping_location"
                ]
            )

            vlm_selected_centroid_xy = np.asarray(
                vlm_result["selected_properties"][
                    "centroid_coordinates"
                ],
                dtype=np.float64,
            ).reshape(2)

            print("VLM selected object:", vlm_selected_object)
            print(
                "VLM selected centroid pixel xy:",
                vlm_selected_centroid_xy,
            )

            print(
                "Simulator GT target (NOT sent to VLM/DINO/GraspNet):",
                target_object,
            )
            print(
                "Preferred grasping location:",
                preferred_grasping_location,
            )
            print("VLM raw output:")
            print(vlm_result["raw_output"])

            VLM_SELECTION_OUTPUT_DIR.mkdir(
                parents=True,
                exist_ok=True,
            )
            vlm_viz_path = (
                VLM_SELECTION_OUTPUT_DIR
                / (
                    f"{target_object}_attempt_{attempt}_"
                    "vlm.png"
                )
            )
            save_vlm_selection_visualization(
                image=topview_data["color"],
                bbox_xyxy=vlm_result["result"]["cropping_box"],
                selected_object=vlm_selected_object,
                preferred_location=preferred_grasping_location,
                output_path=vlm_viz_path,
            )
            print(
                "VLM selection visualization saved:",
                vlm_viz_path,
            )

            # GroundingDINO receives the object name selected by Qwen.
            # The internal MuJoCo target name is resolved separately above.
            grounding_prompt = (
                vlm_selected_object
                .replace("_", " ")
            )

            print(
                "GroundingDINO prompt (direct from VLM):",
                grounding_prompt,
            )

            grounding_result = run_groundingdino_detection(
                image=topview_data["color"],
                text_prompt=grounding_prompt,
            )

            raw_boxes = grounding_result["boxes"]
            raw_scores = grounding_result["scores"]
            raw_phrases = grounding_result["phrases"]

            print(
                "GroundingDINO raw candidate count:",
                len(raw_boxes),
            )
            print(
                "GroundingDINO allowed perception crop:",
                np.array(
                    [
                        0,
                        0,
                        topview_data["image_width"],
                        topview_data["image_height"],
                    ],
                    dtype=np.int32,
                ),
            )

            workspace_filtered = (
                filter_grounding_candidates_to_perception_workspace(
                    boxes=raw_boxes,
                    scores=raw_scores,
                    phrases=raw_phrases,
                    crop_xyxy=np.array(
                        [
                            0,
                            0,
                            topview_data["image_width"],
                            topview_data["image_height"],
                        ],
                        dtype=np.int32,
                    ),
                )
            )

            if len(raw_boxes) > 0:
                for raw_index, (
                    raw_box,
                    raw_score,
                    raw_phrase,
                    raw_center,
                    keep,
                ) in enumerate(
                    zip(
                        raw_boxes,
                        raw_scores,
                        raw_phrases,
                        workspace_filtered["centres_xy"],
                        workspace_filtered["keep_mask"],
                    )
                ):
                    print(
                        "GroundingDINO raw candidate "
                        f"{raw_index}: phrase={str(raw_phrase)!r}, "
                        f"score={float(raw_score):.4f}, "
                        f"bbox={np.asarray(raw_box)}, "
                        f"center={np.round(raw_center, 2)}, "
                        f"inside_purple={bool(keep)}"
                    )

            boxes = workspace_filtered["boxes"]
            scores = workspace_filtered["scores"]
            phrases = workspace_filtered["phrases"]

            print(
                "GroundingDINO candidates after purple-workspace filter:",
                len(boxes),
            )

            if len(boxes) == 0:
                print(
                    "GroundingDINO found no target bbox inside the "
                    "purple perception workspace. Re-perceiving the scene."
                )
                continue

            # VLM-centroid-guided GroundingDINO disambiguation.
            # Primary criterion: bbox centre nearest to the VLM-selected
            # object's centroid. Secondary criterion: higher DINO score.
            dino_centres_xy = np.column_stack(
                (
                    (boxes[:, 0] + boxes[:, 2]) / 2.0,
                    (boxes[:, 1] + boxes[:, 3]) / 2.0,
                )
            )

            vlm_centroid_distances_px = np.linalg.norm(
                dino_centres_xy
                - vlm_selected_centroid_xy[None, :],
                axis=1,
            )

            candidate_order = np.lexsort(
                (
                    -scores,
                    vlm_centroid_distances_px,
                )
            )

            print(
                "VLM-centroid-guided GroundingDINO candidate ranking:"
            )

            for rank, ranked_index in enumerate(
                candidate_order,
                start=1,
            ):
                ranked_index = int(ranked_index)
                print(
                    f"  rank {rank}: "
                    f"index={ranked_index}, "
                    f"center="
                    f"{np.round(dino_centres_xy[ranked_index], 2)}, "
                    f"distance_px="
                    f"{vlm_centroid_distances_px[ranked_index]:.2f}, "
                    f"score={float(scores[ranked_index]):.4f}"
                )

            selected_candidate_index = None
            selected_scene_result = None

            for candidate_rank, candidate_index in enumerate(
                candidate_order,
                start=1,
            ):
                candidate_index = int(candidate_index)
                candidate_box = boxes[candidate_index]
                candidate_score = float(
                    scores[candidate_index]
                )
                candidate_phrase = str(
                    phrases[candidate_index]
                )

                print()
                print(
                    f"Checking GroundingDINO candidate "
                    f"{candidate_rank}/{len(candidate_order)}"
                )
                print("Candidate phrase:", candidate_phrase)
                print("Candidate score:", candidate_score)
                print("Candidate bbox xyxy:", candidate_box)

                try:
                    candidate_scene_result = (
                        export_pybullet_style_target_scene(
                            output_path=SCENE_PATH,
                            heightmap_data=topview_data,
                            bbox_xyxy=candidate_box,
                            workspace_limits=(
                                env.perception_workspace_limits
                            ),
                            crop_margin=GROUNDING_CROP_MARGIN,
                        )
                    )
                except RuntimeError as candidate_error:
                    print(
                        "Candidate rejected:",
                        str(candidate_error),
                    )
                    continue

                target_point_count = int(
                    candidate_scene_result[
                        "point_count"
                    ]
                )

                print(
                    "PyBullet-style expanded bbox xyxy:",
                    candidate_scene_result[
                        "expanded_bbox_xyxy"
                    ],
                )
                print(
                    "Positive heightmap pixels inside bbox:",
                    candidate_scene_result[
                        "positive_bbox_pixel_count"
                    ],
                )
                print(
                    "PyBullet-style full workspace-grid points:",
                    target_point_count,
                )
                print(
                    "Measured bbox XYZ min:",
                    candidate_scene_result[
                        "measured_xyz_min"
                    ],
                )
                print(
                    "Measured bbox XYZ max:",
                    candidate_scene_result[
                        "measured_xyz_max"
                    ],
                )

                if (
                    candidate_scene_result[
                        "positive_bbox_pixel_count"
                    ]
                    < MIN_WORKSPACE_POINTS
                ):
                    print(
                        "Candidate rejected because too few positive "
                        "heightmap pixels remained inside the bbox."
                    )
                    continue

                selected_candidate_index = candidate_index
                selected_scene_result = (
                    candidate_scene_result
                )

                print(
                    "Candidate accepted by PyBullet-style "
                    "heightmap target-crop filter."
                )
                break

            if selected_candidate_index is None:
                print(
                    "No GroundingDINO candidate produced a valid "
                    "PyBullet-style target crop. Re-perceiving."
                )
                continue

            scene_result = selected_scene_result
            selected_box = boxes[selected_candidate_index]

            print()
            print(
                "Selected GroundingDINO phrase:",
                str(phrases[selected_candidate_index]),
            )
            print(
                "Selected GroundingDINO score:",
                float(scores[selected_candidate_index]),
            )
            print(
                "Selected GroundingDINO bbox xyxy:",
                selected_box,
            )
            print(
                "Exported PyBullet-style workspace-grid points:",
                scene_result["point_count"],
            )

            preferred_point_result = (
                preferred_location_to_world_point(
                    topview_data=topview_data,
                    bbox_xyxy=selected_box,
                    preferred_location=(
                        preferred_grasping_location
                    ),
                )
            )

            preferred_world_point = (
                preferred_point_result[
                    "world_point"
                ]
            )

            print()
            print(
                "Preferred grasping cell:",
                preferred_point_result[
                    "preferred_location"
                ],
            )
            print(
                "Preferred cell xyxy:",
                preferred_point_result[
                    "cell_xyxy"
                ],
            )
            print(
                "Preferred requested cell-center pixel xy:",
                preferred_point_result[
                    "requested_center_pixel_xy"
                ],
            )
            print(
                "Preferred selected valid pixel xy:",
                preferred_point_result[
                    "center_pixel_xy"
                ],
            )
            print(
                "Preferred-point fallback mode:",
                preferred_point_result[
                    "fallback_mode"
                ],
            )
            print(
                "Preferred world point xyz:",
                preferred_world_point,
            )

            GROUNDING_GRID_OUTPUT_DIR.mkdir(
                parents=True,
                exist_ok=True,
            )
            grounding_grid_viz_path = (
                GROUNDING_GRID_OUTPUT_DIR
                / (
                    f"{target_object}_attempt_{attempt}_"
                    "grounding_grid.png"
                )
            )
            save_grounding_grid_visualization(
                image=topview_data["color"],
                bbox_xyxy=selected_box,
                preferred_location=preferred_grasping_location,
                center_pixel_xy=preferred_point_result[
                    "center_pixel_xy"
                ],
                output_path=grounding_grid_viz_path,
            )
            print(
                "Grounding 3x3 visualization saved:",
                grounding_grid_viz_path,
            )

            # Diagnostic RGB snapshots. The saved topview image is the
            # orthographic heightmap actually used by VLM / DINO / target
            # crop planning; the other images remain raw camera views.
            PERCEPTION_VIEW_OUTPUT_DIR.mkdir(
                parents=True,
                exist_ok=True,
            )

            perception_view_configs = (
                ("topview", 640, 640),
                ("frontview", 640, 480),
                ("left_oblique_25deg", 640, 480),
                ("right_oblique_25deg", 640, 480),
            )

            for (
                diagnostic_camera_name,
                diagnostic_width,
                diagnostic_height,
            ) in perception_view_configs:
                if diagnostic_camera_name == "topview":
                    diagnostic_camera_data = topview_data
                else:
                    diagnostic_camera_data = env.get_camera_data(
                        camera_name=diagnostic_camera_name,
                        width=diagnostic_width,
                        height=diagnostic_height,
                    )

                diagnostic_image_path = (
                    PERCEPTION_VIEW_OUTPUT_DIR
                    / (
                        f"{target_object}_attempt_{attempt}_"
                        f"{diagnostic_camera_name}.png"
                    )
                )

                diagnostic_image = np.asarray(
                    diagnostic_camera_data["color"]
                ).copy()

                # robosuite's built-in frontview is vertically inverted
                # relative to the human-readable image convention used by
                # the custom top / symmetric oblique cameras.
                if diagnostic_camera_name in (
                    "frontview",
                    "left_oblique_25deg",
                ):
                    diagnostic_image = np.flipud(
                        diagnostic_image
                    ).copy()

                imageio.imwrite(
                    diagnostic_image_path,
                    diagnostic_image,
                )

                print(
                    "Perception view saved:",
                    diagnostic_image_path,
                )

            # Export exactly the points and colors stored in the
            # accepted NPZ scene that will be passed to GraspNet.
            FUSED_CLOUD_PREVIEW_DIR.mkdir(
                parents=True,
                exist_ok=True,
            )

            fused_cloud_ply_path = (
                FUSED_CLOUD_PREVIEW_DIR
                / f"{target_object}_attempt_{attempt}.ply"
            )

            with np.load(
                SCENE_PATH,
                allow_pickle=False,
            ) as scene_data:
                exported_ply_path = (
                    export_colored_pointcloud_ply(
                        output_path=fused_cloud_ply_path,
                        points=scene_data["points"],
                        colors=scene_data["colors"],
                    )
                )

            print(
                "GraspNet input PLY saved:",
                exported_ply_path,
            )

            run_graspnet_inference(
                input_path=SCENE_PATH,
                output_path=GRASP_PATH,
                cuda_visible_devices="0",
            )

            grasp_data = load_target_grasp_data(
                GRASP_PATH
            )

            full_scene_debug_points, full_scene_debug_colors = (
                prepare_full_scene_grasp_debug_background(
                    env=env,
                )
            )

            visualization_all_grasps = np.asarray(
                grasp_data["grasps"],
                dtype=np.float64,
            ).reshape(-1, 7).copy()

            crop_assigned = (
                assign_grasps_to_objects_pybullet_style(
                    env=env,
                    grasps=grasp_data["grasps"],
                    scores=grasp_data["scores"],
                    angles_deg=grasp_data["angles_deg"],
                )
            )

            candidate_grasps = crop_assigned["grasps"]
            candidate_scores = crop_assigned["scores"]
            candidate_angles = crop_assigned["angles_deg"]
            candidate_object_names = (
                crop_assigned["grasp_object_names"]
            )
            candidate_object_body_ids = (
                crop_assigned["grasp_object_body_ids"]
            )
            candidate_object_distances = (
                crop_assigned["grasp_object_distances"]
            )

            print(
                "PyBullet-style object-assignment distance threshold:",
                crop_assigned["distance_threshold"],
            )
            print(
                "Target-crop grasps assigned to ANY simulator object:",
                len(candidate_grasps),
            )

            assigned_object_summary = {}
            for assigned_name in candidate_object_names:
                assigned_name = str(assigned_name)
                assigned_object_summary[assigned_name] = (
                    assigned_object_summary.get(assigned_name, 0) + 1
                )

            print(
                "Target-crop assigned-grasp counts by object:",
                assigned_object_summary,
            )

            # Diagnostic only: retain the fixed GT target-distance printout,
            # but it no longer filters the grasp-selection pool.
            target_gt_center = np.asarray(
                env.sim.data.body_xpos[target_body_id],
                dtype=np.float64,
            ).copy()

            all_crop_grasps = np.asarray(
                grasp_data["grasps"],
                dtype=np.float64,
            ).reshape(-1, 7)

            crop_gt_distances = np.linalg.norm(
                all_crop_grasps[:, :3]
                - target_gt_center[None, :],
                axis=1,
            )

            print()
            print("GT-ASSIGNMENT DIAGNOSTIC — TARGET CROP")
            print("Target GT body center xyz:", target_gt_center)

            if len(crop_gt_distances) > 0:
                crop_distance_order = np.argsort(crop_gt_distances)

                print(
                    "Nearest target-crop grasp distance to GT center [m]:",
                    float(crop_gt_distances[crop_distance_order[0]]),
                )
                print(
                    "Target-crop grasps within 5 cm of GT center:",
                    int(np.count_nonzero(
                        crop_gt_distances
                        < PYBULLET_GRASP_OBJECT_DISTANCE_M
                    )),
                )

            used_full_scene_fallback = False

            if len(candidate_grasps) == 0:
                print()
                print(
                    "Target crop produced 0 object-assigned grasps. "
                    "Running source-style full-scene fallback."
                )

                run_graspnet_inference(
                    input_path=FULL_SCENE_PATH,
                    output_path=FULL_GRASP_PATH,
                    cuda_visible_devices="0",
                )

                full_grasp_data = (
                    load_target_grasp_data(
                        FULL_GRASP_PATH
                    )
                )

                print(
                    "Full-scene fallback collision-filtered GraspNet grasps:",
                    len(full_grasp_data["grasps"]),
                )

                visualization_all_grasps = np.asarray(
                    full_grasp_data["grasps"],
                    dtype=np.float64,
                ).reshape(-1, 7).copy()

                fallback_assigned = (
                    assign_grasps_to_objects_pybullet_style(
                        env=env,
                        grasps=full_grasp_data["grasps"],
                        scores=full_grasp_data["scores"],
                        angles_deg=full_grasp_data["angles_deg"],
                    )
                )

                candidate_grasps = fallback_assigned["grasps"]
                candidate_scores = fallback_assigned["scores"]
                candidate_angles = fallback_assigned["angles_deg"]
                candidate_object_names = (
                    fallback_assigned["grasp_object_names"]
                )
                candidate_object_body_ids = (
                    fallback_assigned["grasp_object_body_ids"]
                )
                candidate_object_distances = (
                    fallback_assigned["grasp_object_distances"]
                )

                print(
                    "Full-scene grasps assigned to ANY simulator object:",
                    len(candidate_grasps),
                )

                if len(candidate_grasps) == 0:
                    print(
                        "Full-scene fallback produced no grasp assigned "
                        "to any simulator object by the original PyBullet "
                        "5 cm nearest-object rule."
                    )
                    print(
                        "PyBullet-style behavior: ending this case instead "
                        "of re-perceiving an unchanged scene."
                    )
                    ended_no_grasp_after_fallback = True
                    break

                used_full_scene_fallback = True

                print(
                    "Using all PyBullet-assigned full-scene grasps "
                    "for final preferred-location selection."
                )

            all_grasps_ply_path = (
                save_grasp_set_debug_ply(
                    env=env,
                    grasps=visualization_all_grasps,
                    scene_points=full_scene_debug_points,
                    scene_colors=full_scene_debug_colors,
                    output_dir=GRASP_DEBUG_ALL_GRASPS_DIR,
                    output_name=(
                        f"{target_object}_attempt_{attempt}_"
                        "all_grasps.ply"
                    ),
                )
            )

            target_assigned_ply_path = (
                save_grasp_set_debug_ply(
                    env=env,
                    grasps=candidate_grasps,
                    scene_points=full_scene_debug_points,
                    scene_colors=full_scene_debug_colors,
                    output_dir=GRASP_DEBUG_TARGET_ASSIGNED_DIR,
                    output_name=(
                        f"{target_object}_attempt_{attempt}_"
                        "target_assigned.ply"
                    ),
                )
            )

            print(
                "All grasps full-scene PLY saved:",
                all_grasps_ply_path,
            )
            print(
                "Target-assigned full-scene PLY saved:",
                target_assigned_ply_path,
            )

            # Diagnostic only: show every object-assigned grasp before
            # source-faithful 15-degree / preferred-XY selection.
            all_candidate_xy_distances = np.linalg.norm(
                candidate_grasps[:, :2]
                - preferred_world_point[None, :2],
                axis=1,
            )

            print()
            print("All PyBullet-assigned grasps before final selection:")
            print(
                "index | object | score | xy_distance_m | approach_angle_deg"
            )
            for candidate_index in range(len(candidate_grasps)):
                print(
                    f"{candidate_index:5d} | "
                    f"{str(candidate_object_names[candidate_index]):18s} | "
                    f"{float(candidate_scores[candidate_index]):.6f} | "
                    f"{float(all_candidate_xy_distances[candidate_index]):.6f} | "
                    f"{float(candidate_angles[candidate_index]):.3f}"
                )

            print(
                "Grasp source for final selection:",
                (
                    "source-style four-view ICP full-scene fallback"
                    if used_full_scene_fallback
                    else "PyBullet-style heightmap target crop"
                ),
            )

            print()
            print(
                "Source-style execution policy: selecting exactly one grasp "
                "for this perception cycle."
            )
            print(
                "The fixed testcase GT target does NOT pre-filter this pool."
            )

            selection = select_grasp_pybullet_style(
                grasps=candidate_grasps,
                scores=candidate_scores,
                angles_deg=candidate_angles,
                object_names=candidate_object_names,
                preferred_world_point=(
                    preferred_world_point
                ),
                angle_threshold_deg=(
                    PYBULLET_GRASP_ANGLE_DEG
                ),
            )

            selected_grasp = selection[
                "selected_grasp"
            ]

            print()
            print(
                "PyBullet angle threshold degrees:",
                selection["angle_threshold_deg"],
            )
            print(
                "PyBullet ACTUAL object-wise angle filtering:"
            )
            for object_filter in selection[
                "per_object_angle_filter"
            ]:
                print(
                    "  object="
                    f"{object_filter['object_name']}, "
                    "assigned_indices="
                    f"{object_filter['assigned_indices'].tolist()}, "
                    "safe_indices="
                    f"{object_filter['safe_indices'].tolist()}, "
                    "kept_indices="
                    f"{object_filter['kept_indices'].tolist()}, "
                    "mode="
                    f"{object_filter['mode']}"
                )
            print(
                "Merged selection-pool indices:",
                selection["selection_pool_indices"],
            )
            print(
                "Merged selection-pool object names:",
                selection["selection_pool_object_names"],
            )
            print(
                "Selection-pool XY distances:",
                selection["selection_pool_xy_distances"],
            )
            print(
                "Selection-pool scores (diagnostic only):",
                selection["selection_pool_scores"],
            )
            print(
                "Selection-pool angles degrees:",
                selection["selection_pool_angles_deg"],
            )
            selected_grasp_index = int(
                selection["selected_index"]
            )

            selected_assigned_object_name = str(
                candidate_object_names[selected_grasp_index]
            )
            selected_assigned_object_body_id = int(
                candidate_object_body_ids[selected_grasp_index]
            )
            selected_assigned_object_distance = float(
                candidate_object_distances[selected_grasp_index]
            )

            print(
                "Selected grasp assigned simulator object:",
                selected_assigned_object_name,
            )
            print(
                "Selected grasp assigned simulator body id:",
                selected_assigned_object_body_id,
            )
            print(
                "Selected grasp assignment distance [m]:",
                selected_assigned_object_distance,
            )

            print(
                "Selected grasp candidate index:",
                selected_grasp_index,
            )
            print(
                "Selected grasp score:",
                selection[
                    "selected_score"
                ],
            )
            print(
                "Selected grasp XY distance:",
                selection[
                    "selected_xy_distance"
                ],
            )
            print(
                "Selected grasp:",
                selected_grasp,
            )

            selected_grasp_angle_deg = float(
                selection["selected_angle_deg"]
            )

            # Diagnostic only: reproduce the pose conversion used by
            # execute_grasp_pose_ik() so we can inspect the selected target
            # before any motion is commanded. PyBullet-style over is WORLD
            # +Z by 0.20 m while preserving the final grasp orientation.
            selected_grasp_eef_pose = (
                env.grasp_tip_pose_to_eef_pose(
                    selected_grasp
                )
            )
            selected_approach_axis = (
                selected_grasp_eef_pose[:3, 2]
            ).copy()

            predicted_pregrasp_pose = (
                selected_grasp_eef_pose.copy()
            )
            predicted_pregrasp_pose[2, 3] += 0.20

            print()
            print(
                "Selected grasp angle degrees:",
                selected_grasp_angle_deg,
            )
            print(
                "Selected grasp center xyz:",
                selected_grasp[:3],
            )
            print(
                "Selected grasp quaternion xyzw:",
                selected_grasp[3:],
            )
            print(
                "Selected grasp EEF target position xyz:",
                selected_grasp_eef_pose[:3, 3],
            )
            print(
                "Selected grasp EEF target rotation matrix:",
            )
            print(
                selected_grasp_eef_pose[:3, :3]
            )
            print(
                "Selected approach axis:",
                selected_approach_axis,
            )
            print(
                "Predicted PyBullet-style world-Z over target xyz:",
                predicted_pregrasp_pose[:3, 3],
            )
            print(
                "Predicted pregrasp target rotation matrix:",
            )
            print(
                predicted_pregrasp_pose[:3, :3]
            )

            selected_grasp_ply_path = (
                save_grasp_set_debug_ply(
                    env=env,
                    grasps=np.asarray(
                        selected_grasp,
                        dtype=np.float64,
                    ).reshape(1, 7),
                    scene_points=full_scene_debug_points,
                    scene_colors=full_scene_debug_colors,
                    output_dir=GRASP_DEBUG_SELECTED_GRASP_DIR,
                    output_name=(
                        f"{target_object}_attempt_{attempt}_"
                        "selected_grasp.ply"
                    ),
                )
            )

            print(
                "Selected grasp full-scene PLY saved:",
                selected_grasp_ply_path,
            )

            if GRASP_CONTROL_MODE == "ik":
                print()
                print(
                    "Executing selected grasp with "
                    "IK + q_ref + JOINT_POSITION."
                )

                execution = env.execute_grasp_pose_ik(
                    selected_grasp,
                    pregrasp_height=0.20,
                    lift_distance=0.20,
                    grasp_depth_offset=0.0,
                    waypoint_spacing=0.01,
                    qref_speed_rad_per_sec=1.0,
                    frame_callback=(
                        lambda _env: recorder.capture_frame()
                    ),
                    frame_capture_interval=(
                        IK_VIDEO_PHYSICS_CAPTURE_INTERVAL
                    ),
                    stop_after_pregrasp=(
                        IK_STOP_AFTER_PREGRASP
                    ),
                    stop_after_grasp_pose=(
                        IK_STOP_AFTER_GRASP_POSE
                    ),
                )
            else:
                print()
                print(
                    "Executing selected grasp with legacy OSC."
                )

                execution = env.execute_grasp_pose(
                    selected_grasp,
                    approach_distance=0.12,
                    lift_distance=0.10,
                    grasp_depth_offset=0.0,
                    # ThinkGrasp原版move_joints超时为3秒。
                    # 当前MuJoCo控制频率约20Hz，因此使用60步。
                    max_steps_per_phase=120,
                    position_tolerance=0.008,
                    orientation_tolerance=np.deg2rad(4.0),
                )

            # ---------------------------------------------------------
            # Staged IK integration stop.
            #
            # This is a SUCCESSFUL diagnostic termination, not a failed
            # grasp. Do not run grasp-success inference, placement, or the
            # legacy OSC recovery functions while the environment is using
            # JOINT_POSITION.
            # ---------------------------------------------------------
            diagnostic_stage = execution.get(
                "stopped_for_diagnostic"
            )

            if diagnostic_stage is not None:
                diagnostic_completed = True
                diagnostic_stop_stage = str(
                    diagnostic_stage
                )

                actual_pose = (
                    env.get_eef_pose().copy()
                )

                print()
                print("#" * 70)
                print(
                    "IK STAGED INTEGRATION DIAGNOSTIC COMPLETED"
                )
                print("#" * 70)
                print(
                    "Diagnostic stop stage:",
                    diagnostic_stop_stage,
                )
                print(
                    "Execution success:",
                    execution.get("success"),
                )
                print(
                    "Failed phase:",
                    execution.get("failed_phase"),
                )
                print(
                    "Actual EEF xyz at diagnostic stop:",
                    actual_pose[:3, 3],
                )

                returned_pregrasp_pose = (
                    execution.get(
                        "pregrasp_pose"
                    )
                )

                if returned_pregrasp_pose is not None:
                    returned_pregrasp_pose = np.asarray(
                        returned_pregrasp_pose,
                        dtype=np.float64,
                    ).reshape(4, 4)

                    print(
                        "Returned pregrasp target xyz:",
                        returned_pregrasp_pose[:3, 3],
                    )
                    print(
                        "Pregrasp target minus actual xyz:",
                        (
                            returned_pregrasp_pose[:3, 3]
                            - actual_pose[:3, 3]
                        ),
                    )

                safe_rest_result = execution.get(
                    "safe_rest"
                )
                if safe_rest_result is not None:
                    print(
                        "Safe-rest motion success:",
                        safe_rest_result.get("success"),
                    )
                    print(
                        "Safe-rest max joint error [rad]:",
                        safe_rest_result.get(
                            "max_abs_joint_error"
                        ),
                    )
                    print(
                        "Safe-rest simulated seconds:",
                        safe_rest_result.get(
                            "simulated_seconds"
                        ),
                    )

                pregrasp_result = execution.get(
                    "pregrasp"
                )
                if pregrasp_result is not None:
                    print(
                        "Pregrasp success:",
                        pregrasp_result.get("success"),
                    )
                    print(
                        "Pregrasp failure reason:",
                        pregrasp_result.get(
                            "failure_reason"
                        ),
                    )
                    print(
                        "Pregrasp position error [m]:",
                        pregrasp_result.get(
                            "position_error_norm"
                        ),
                    )
                    print(
                        "Pregrasp orientation error [deg]:",
                        pregrasp_result.get(
                            "orientation_error_deg"
                        ),
                    )

                recorder.capture_frame()
                recorder.add_hold(2.0)

                print(
                    "Diagnostic stage complete. "
                    "Stopping before descent / close / lift."
                )
                break

            # ---------------------------------------------------------
            # STRICT-IK FAILURE RECOVERY (JOINT_POSITION ONLY)
            #
            # If the selected grasp is not IK-converged / executable,
            # reject this perception cycle, return to the saved home joint
            # configuration using q_ref + JOINT_POSITION, then re-perceive.
            # No legacy OSC recovery is used.
            # ---------------------------------------------------------
            if (
                GRASP_CONTROL_MODE == "ik"
                and not bool(execution.get("success", False))
            ):
                failed_phase = execution.get("failed_phase")

                print()
                print("#" * 70)
                print("IK EXECUTION FAILED — RECOVERING HOME")
                print("#" * 70)
                print("Failed phase:", failed_phase)

                # Diagnostic only: expose why the selected pregrasp IK was
                # rejected. This does not change IK thresholds or execution.
                if failed_phase == "pregrasp":
                    pregrasp_result = execution.get(
                        "pregrasp"
                    )

                    if isinstance(pregrasp_result, dict):
                        ik_diagnostic = pregrasp_result.get(
                            "ik"
                        )

                        if isinstance(ik_diagnostic, dict):
                            position_error_mm = (
                                1000.0
                                * float(
                                    ik_diagnostic.get(
                                        "position_error_norm",
                                        float("nan"),
                                    )
                                )
                            )

                            orientation_error_deg = float(
                                ik_diagnostic.get(
                                    "orientation_error_deg",
                                    float("nan"),
                                )
                            )

                            print()
                            print(
                                "PREGRASP IK DIAGNOSTIC"
                            )
                            print(
                                "Position error:",
                                position_error_mm,
                                "mm",
                            )
                            print(
                                "Orientation error:",
                                orientation_error_deg,
                                "deg",
                            )
                            print(
                                "Iterations:",
                                ik_diagnostic.get(
                                    "iterations"
                                ),
                            )
                            print(
                                "Converged:",
                                ik_diagnostic.get(
                                    "converged"
                                ),
                            )
                            print(
                                "Usable solution:",
                                ik_diagnostic.get(
                                    "usable_solution"
                                ),
                            )
                            print(
                                "Residual threshold:",
                                ik_diagnostic.get(
                                    "residual_threshold"
                                ),
                            )
                            print(
                                "Best-effort joint positions:",
                                ik_diagnostic.get(
                                    "joint_positions"
                                ),
                            )

                # Before close/lift, recover with the gripper open.
                # If failure happened after closing, keep holding during the
                # home motion and release only after home is reached.
                hold_during_recovery = bool(
                    failed_phase in {"close", "lift"}
                )

                recovery_result = env.move_joints_qref(
                    target_joint_positions=home_joint_positions,
                    gripper_command=(
                        1.0 if hold_during_recovery else -1.0
                    ),
                    stop_on_table_contact=False,
                    frame_callback=(
                        lambda _env: recorder.capture_frame()
                    ),
                    frame_capture_interval=IK_VIDEO_PHYSICS_CAPTURE_INTERVAL,
                )

                print(
                    "IK/q_ref return-home success:",
                    recovery_result["success"],
                )
                print(
                    "IK/q_ref return-home failure reason:",
                    recovery_result["failure_reason"],
                )

                if recovery_result["success"]:
                    env.open_gripper(steps=30)
                    recorder.capture_frame()
                    recorder.add_hold(0.5)

                    print(
                        "Recovery complete. Starting a new perception / "
                        "planning cycle."
                    )
                    continue

                print(
                    "Return-home recovery failed. Stopping this run without "
                    "calling any legacy OSC controller."
                )
                recorder.capture_frame()
                recorder.add_hold(2.0)
                break

            # =========================================================
            # Source-faithful grasp / transport task semantics.
            #
            # Physical grasp success and task-target success are separate:
            #   - gripper width answers "are we still holding something?"
            #   - selected 5 cm assignment records which simulator object this
            #     grasp belongs to for post-transport task evaluation.
            #
            # Any held object is transported to the bin. Only AFTER release
            # and JOINT_POSITION return-home do we decide target vs non-target.
            # =========================================================
            first_gripper_width = env.get_gripper_width()

            gripper_holds_something = bool(
                execution["success"]
                and first_gripper_width
                >= MIN_GRASPED_GRIPPER_WIDTH
            )

            print()
            print("Motion success:", execution["success"])
            print(
                "Gripper width after lift:",
                first_gripper_width,
            )
            print(
                "First gripper-width hold check:",
                gripper_holds_something,
            )
            print(
                "Selected grasp assigned object:",
                selected_assigned_object_name,
            )
            print(
                "Selected grasp assigned body id:",
                selected_assigned_object_body_id,
            )
            print(
                "Fixed testcase target object:",
                target_object,
            )
            print(
                "Fixed testcase target body id:",
                target_body_id,
            )

            for phase_name in [
                "pregrasp",
                "grasp",
                "lift",
            ]:
                phase_result = execution.get(phase_name)

                if phase_result is None:
                    continue

                print()
                print(
                    f"{phase_name} success:",
                    phase_result.get("success"),
                )
                print(
                    f"{phase_name} failure reason:",
                    phase_result.get("failure_reason"),
                )
                print(
                    f"{phase_name} position error:",
                    phase_result.get("position_error_norm"),
                )

                phase_diagnostic = (
                    summarize_ik_phase_diagnostic(
                        phase_result
                    )
                )

                print(
                    f"{phase_name} diagnostic motion segments:",
                    phase_diagnostic["motion_segments"],
                )
                print(
                    f"{phase_name} diagnostic physics steps:",
                    phase_diagnostic["physics_steps"],
                )
                print(
                    f"{phase_name} diagnostic simulated seconds:",
                    phase_diagnostic["simulated_seconds"],
                )
                print(
                    f"{phase_name} diagnostic max tracking error [rad]:",
                    phase_diagnostic["max_tracking_error_seen"],
                )
                print(
                    f"{phase_name} diagnostic max raw torque:",
                    phase_diagnostic["max_raw_torque_seen"],
                )
                print(
                    f"{phase_name} diagnostic max relevant contacts/step:",
                    phase_diagnostic["max_relevant_contact_count"],
                )
                print(
                    f"{phase_name} diagnostic contact pairs:",
                    phase_diagnostic["diagnostic_contact_pairs"],
                )
                print(
                    f"{phase_name} source-style max EEF force metric:",
                    phase_diagnostic["max_eef_force_metric_seen"],
                )
                print(
                    f"{phase_name} force-stop triggered:",
                    phase_diagnostic["force_stop_triggered"],
                )
                if phase_diagnostic["force_stop_triggered"]:
                    print(
                        f"{phase_name} force-stop metric:",
                        phase_diagnostic["force_stop_metric"],
                    )
                    print(
                        f"{phase_name} force-stop wrench [Fx Fy Fz Mx My Mz]:",
                        phase_diagnostic["force_stop_wrench"],
                    )

            # Physical grasp failed: no object is retained by the gripper.
            # Formal IK mode uses JOINT_POSITION recovery only.
            if not gripper_holds_something:
                print()
                print(
                    "Physical grasp failed: gripper-width check indicates "
                    "that no object is being held."
                )

                recovery_result = env.move_joints_qref(
                    target_joint_positions=home_joint_positions,
                    gripper_command=-1.0,
                    stop_on_table_contact=False,
                    frame_callback=(
                        lambda _env: recorder.capture_frame()
                    ),
                    frame_capture_interval=(
                        IK_VIDEO_PHYSICS_CAPTURE_INTERVAL
                    ),
                )

                print(
                    "JOINT_POSITION return home after empty grasp success:",
                    recovery_result["success"],
                )

                if not recovery_result["success"]:
                    print(
                        "Return-home recovery failed. Stopping without "
                        "legacy OSC recovery."
                    )
                    break

                env.open_gripper(steps=30)
                recorder.capture_frame()
                recorder.add_hold(0.5)

                print(
                    "Home reached. Re-perceiving the current scene."
                )
                continue

            print()
            print(
                "Physical grasp succeeded. Transporting the held object "
                "before evaluating target vs non-target."
            )

            # =========================================================
            # PyBullet-style fixed joint-space transport.
            #
            # Original ThinkGrasp moves the held object to a fixed
            # drop joint configuration after lift. Do the same here:
            # no workspace-centre IK, no straighten IK, no bin IK.
            # =========================================================
            print()
            print(
                "Starting PyBullet-style fixed JOINT_POSITION transport."
            )

            recorder.capture_frame()
            recorder.add_hold(0.5)

            print(
                "Fixed Panda drop joints:",
                PANDA_DROP_JOINTS,
            )

            drop_motion_result = env.move_joints_qref(
                target_joint_positions=PANDA_DROP_JOINTS,
                gripper_command=1.0,
                stop_on_table_contact=False,
                frame_callback=(
                    lambda _env: recorder.capture_frame()
                ),
                frame_capture_interval=(
                    IK_VIDEO_PHYSICS_CAPTURE_INTERVAL
                ),
            )

            print(
                "Fixed drop-joint transport success:",
                drop_motion_result["success"],
            )
            print(
                "Fixed drop-joint transport failure reason:",
                drop_motion_result["failure_reason"],
            )

            if not drop_motion_result["success"]:
                print(
                    "Fixed drop-joint transport failed."
                )
                break

            # ---------------------------------------------------------
            # Second original-style gripper-state check:
            # after reaching the drop/bin pose, immediately before release.
            # ---------------------------------------------------------
            width_before_release = env.get_gripper_width()
            still_holding_at_bin = bool(
                width_before_release
                >= MIN_GRASPED_GRIPPER_WIDTH
            )

            print(
                "Gripper width at bin before release:",
                width_before_release,
            )
            print(
                "Second gripper-width hold check:",
                still_holding_at_bin,
            )

            if not still_holding_at_bin:
                print(
                    "Held object was lost during transport. Returning home "
                    "without issuing an intentional bin-release command."
                )

                return_home_result = env.move_joints_qref(
                    target_joint_positions=home_joint_positions,
                    gripper_command=-1.0,
                    stop_on_table_contact=False,
                    frame_callback=(
                        lambda _env: recorder.capture_frame()
                    ),
                    frame_capture_interval=(
                        IK_VIDEO_PHYSICS_CAPTURE_INTERVAL
                    ),
                )

                print(
                    "JOINT_POSITION return home after transport loss:",
                    return_home_result["success"],
                )

                if not return_home_result["success"]:
                    break

                env.open_gripper(steps=30)
                recorder.capture_frame()
                recorder.add_hold(0.5)

                print(
                    "Home reached after transport loss. Re-perceiving."
                )
                continue

            # Physical transport succeeded. Release the held object in the bin.
            open_result = env.open_gripper(steps=40)
            width_after_release = env.get_gripper_width()

            print(
                "Gripper width after intentional bin release:",
                width_after_release,
            )
            print(
                "Release command width:",
                open_result["width"],
            )

            recorder.capture_frame()
            recorder.add_hold(1.0)

            # Source-faithful ordering:
            # release -> home -> THEN evaluate target/non-target and re-perceive.
            return_home_result = env.move_joints_qref(
                target_joint_positions=home_joint_positions,
                gripper_command=-1.0,
                stop_on_table_contact=False,
                frame_callback=(
                    lambda _env: recorder.capture_frame()
                ),
                frame_capture_interval=(
                    IK_VIDEO_PHYSICS_CAPTURE_INTERVAL
                ),
            )

            print(
                "JOINT_POSITION return home after release success:",
                return_home_result["success"],
            )

            if not return_home_result["success"]:
                print(
                    "Return home after release failed; stopping without "
                    "legacy OSC recovery."
                )
                break

            moved_target_object = bool(
                selected_assigned_object_name == target_object
                and selected_assigned_object_body_id == target_body_id
            )

            print()
            print(
                "Released object identity from selected 5 cm assignment:",
                selected_assigned_object_name,
            )
            print(
                "Released object is fixed testcase target:",
                moved_target_object,
            )

            if moved_target_object:
                target_completed = True
                print(
                    f"Target {target_object} was transported to the bin, "
                    "released, and the robot returned home."
                )
                break

            print(
                "A non-target object was successfully removed to the bin. "
                "done=False; robot is home, so re-perceiving the changed scene."
            )
            recorder.capture_frame()
            recorder.add_hold(0.5)
            continue

        if diagnostic_completed:
            print()
            print(
                f"Closed-loop diagnostic ended after "
                f"{attempts_started} perception attempt(s)."
            )
            print(
                "Diagnostic result: SUCCESSFUL staged stop at",
                diagnostic_stop_stage,
            )
            print(
                "No descent / close / lift / placement was executed."
            )
        elif ik_failure_stopped_safely:
            print()
            print(
                f"Closed-loop IK diagnostic ended after "
                f"{attempts_started} perception attempt(s)."
            )
            print(
                "Diagnostic result: SAFE STOP after IK failure."
            )
            print(
                "IK failed phase:",
                ik_failure_phase,
            )
            print(
                "Legacy OSC recovery was not executed."
            )
        elif not target_completed:
            print()
            print(
                f"Closed-loop task ended after "
                f"{attempts_started} perception attempt(s)."
            )

            if ended_no_grasp_after_fallback:
                print(
                    "Stop reason: source-style target crop and "
                    "full-scene fallback produced no grasp assigned "
                    "to the fixed GT target."
                )
            elif attempts_started >= MAX_ATTEMPTS:
                print(
                    "Stop reason: maximum perception-attempt limit reached."
                )
            else:
                print(
                    "Stop reason: closed-loop task ended before completion."
                )

            print("Closed-loop task failed.")
        else:
            print()
            print(
                f"Closed-loop task completed for {target_object}."
            )

    finally:
        try:
            if recorder.started:
                recorder.capture_frame()
                recorder.add_hold(2.0)

                video_path = recorder.save()

                print()
                print("Closed-loop video saved:")
                print(video_path)
                print(
                    "Recorded frames:",
                    len(recorder.frames),
                )
        except Exception as video_error:
            print(
                "Video saving failed:",
                repr(video_error),
            )
        finally:
            recorder.stop()
            env.close()


if __name__ == "__main__":
    _run_with_log()

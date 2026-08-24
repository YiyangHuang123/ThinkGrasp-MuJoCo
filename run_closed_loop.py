"""Closed-loop robotic grasping in MuJoCo.

Pipeline:
    language goal
    -> MuJoCo RGB-D perception
    -> Qwen3-VL target selection
    -> GroundingDINO target localization
    -> GraspNet grasp generation
    -> target-region grasp ranking
    -> Panda IK + JOINT_POSITION execution
    -> fixed joint-space transport
    -> simulator-side task evaluation
    -> re-perception and re-planning on failure

If the target region contains no usable grasp, a four-view full-scene
fallback grasp is used to perturb the clutter before the next cycle.
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
from PIL import Image, ImageDraw, ImageFont

from graspnet_bridge import (
    load_target_grasp_data,
    run_graspnet_inference,
)
from dual_view_recorder import DualViewRecorder
from scene_bridge import (
    build_pybullet_style_heightmap,
    export_colored_pointcloud_ply,
    export_pybullet_style_target_scene,
)
from thinkgrasp_minimal_env import (
    GSO_SCENE_OBJECT_SPECS,
    ThinkGraspMinimalEnv,
    load_thinkgrasp_joint_position_controller_config,
)
from vlm_bridge import run_vlm_selection
from perception_viz import (
    save_vlm_selection_visualization,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR

# Fixed test cases.
# Line 1: natural-language goal for VLM.
# Line 2: simulator GT target object name, used ONLY for final success check.
CASE_DIR = SCRIPT_DIR / "cases"
DEFAULT_CASE_PATH = CASE_DIR / "case02_white_ramekin.txt"

BRIDGE_DIR = SCRIPT_DIR / "bridge_data"
SCENE_PATH = BRIDGE_DIR / "closed_loop_scene.npz"
GRASP_PATH = BRIDGE_DIR / "closed_loop_grasps.npz"

# Full-scene fallback artifacts:
# target crop -> 0 grasps -> workspace-filtered full four-view scene.
FULL_SCENE_PATH = BRIDGE_DIR / "closed_loop_full_scene.npz"
FULL_SCENE_RAW_VIEWS_PATH = (
    BRIDGE_DIR / "closed_loop_full_scene_raw_views.npz"
)
FULL_GRASP_PATH = BRIDGE_DIR / "closed_loop_full_scene_grasps.npz"
GROUNDING_IMAGE_PATH = BRIDGE_DIR / "perception_rgb.png"
FULL_PERCEPTION_IMAGE_PATH = BRIDGE_DIR / "topview_full_rgb.png"
GROUNDING_RESULT_PATH = BRIDGE_DIR / "grounding_result.npz"

# Project-local copy of the VLM system prompt.
# This keeps the runner self-contained at runtime.
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
GRASP_DEBUG_TARGET_REGION_DIR = (
    GRASP_DEBUG_PREVIEW_DIR / "target_region"
)
GRASP_DEBUG_SELECTED_GRASP_DIR = (
    GRASP_DEBUG_PREVIEW_DIR / "selected_grasp"
)

LOG_OUTPUT_DIR = (
    CLOSED_LOOP_OUTPUT_DIR / "logs"
)

# Compatibility output directories retained for cleanup.
COMPAT_WORKSPACE_PREVIEW_DIR = (
    SCRIPT_DIR / "workspace_preview"
)
COMPAT_VIEW_TESTS_DIR = (
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
        "and point-cloud fusion."
    )

GROUNDING_PYTHON = Path(
    LEGACY_PERCEPTION_PYTHON
).expanduser().resolve()
GROUNDING_BOX_THRESHOLD = 0.15
GROUNDING_TEXT_THRESHOLD = 0.25
# Outer context margin:
# GraspNet sees this larger region so neighbouring geometry remains
# available for collision-aware grasp generation.
GROUNDING_CROP_MARGIN = 20

# Inner target-grasp margin:
# only grasps whose centres fall inside the original GroundingDINO bbox
# plus this small tolerance are eligible for final grasp selection.
GROUNDING_TARGET_GRASP_MARGIN = 5

# GroundingDINO uses a RAW top-view workspace crop; VLM uses the orthographic heightmap RGB.
PERCEPTION_WIDTH = 640
PERCEPTION_HEIGHT = 640

# Full-scene fallback uses the runtime perception workspace stored in
# env.perception_workspace_limits, shared by perception and grasp planning.
FULL_SCENE_TABLE_HEIGHT_M = 0.855
FULL_SCENE_TABLE_CLEARANCE_M = 0.003
FULL_SCENE_MAX_HEIGHT_ABOVE_TABLE_M = 0.30

FULL_SCENE_CAMERAS = (
    ("topview", 640, 640),
    ("front_oblique_25deg", 640, 480),
    ("left_oblique_25deg", 640, 480),
    ("right_oblique_25deg", 640, 480),
)

# A candidate bbox is retained only when enough of its valid 3D points
# remain inside the calibrated world-coordinate workspace.
MIN_WORKSPACE_POINTS = 30

MAX_ATTEMPTS = 50
MIN_GRASPED_GRIPPER_WIDTH = 0.005

# ---------------------------------------------------------------------------
# Target-grasp ranking policy:
# continuous approach-angle quality + VLM preferred-location weighting.
#
# This ranking operates only inside the DINO target-region grasp pool.
# GraspNet confidence remains diagnostic in normal target-grasp mode.
# ---------------------------------------------------------------------------
GRASP_SELECTION_ANGLE_WEIGHT = 0.60
GRASP_SELECTION_PREFERRED_WEIGHT = 0.40
GRASP_SELECTION_ANGLE_SIGMA_DEG = 30.0
GRASP_SELECTION_PREFERRED_SIGMA_M = 0.05

# Full-scene fallback is intentionally target-agnostic.
# It is used only when the selected target region produces zero usable grasps.
# The purpose is to execute one mechanically suitable grasp that perturbs
# the clutter state before the next perception cycle, avoiding repeated failure
# on an unchanged scene. Preferred-location information is not used here.
FULL_SCENE_FALLBACK_ANGLE_WEIGHT = 0.60
FULL_SCENE_FALLBACK_GRASPNET_WEIGHT = 0.40

# DINO bbox soft ranking: DINO confidence + VLM centroid proximity.
# GroundingDINO remains the primary bbox-ranking signal.
# VLM centroid is used only as a soft spatial disambiguation cue.
# Keeping DINO dominant prevents a weak semantic candidate from winning
# only because it happens to be spatially close to the VLM centroid.
DINO_RANKING_CONFIDENCE_WEIGHT = 0.70
DINO_RANKING_CENTROID_WEIGHT = 0.30
DINO_RANKING_CENTROID_SIGMA_M = 0.08

# q_ref advances at MuJoCo's 500 Hz physics rate. Capture every 50 physics
# steps = one frame every 0.1 s = 10 real-time frames/s, matching the
# DualViewRecorder output fps below.
IK_VIDEO_PHYSICS_CAPTURE_INTERVAL = 50

# Fixed Panda drop posture.
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

# ============================================================
# Case Configuration
# ============================================================

# One fixed default target per scene.
#
# Normal usage:
#     python run_closed_loop.py --scene 5
#
# The runner resolves:
#     5 -> scene05 -> cases/case_scene05_gaming_mouse.txt
#
# --case remains available as an explicit manual override.
DEFAULT_CASE_BY_SCENE = {
    "scene01": "case_scene01_white_ramekin.txt",
    "scene02": "case_scene02_nikon_camera.txt",
    "scene03": "case_scene03_mario_figure.txt",
    "scene04": "case_scene04_baby_car.txt",
    "scene05": "case_scene05_gaming_mouse.txt",
    "scene06": "case_scene06_moisturizer_jar.txt",
    "scene07": "case_scene07_rhino_figure.txt",
    "scene08": "case_scene08_creatine_bottle.txt",
    "scene09": "case_scene09_lion_figure.txt",
    "scene10": "case_scene10_crocodile_toy.txt",
}


def _normalize_scene_name(scene_argument):
    """Accept either 1..10 or scene01..scene10 and return sceneXX."""

    value = str(scene_argument).strip().lower()

    if value.isdigit():
        scene_number = int(value)

        if not 1 <= scene_number <= 10:
            raise ValueError(
                "--scene numeric value must be between 1 and 10, "
                f"got {scene_argument!r}."
            )

        value = f"scene{scene_number:02d}"

    if value not in GSO_SCENE_OBJECT_SPECS:
        raise ValueError(
            f"Unknown scene {scene_argument!r}. "
            "Use 1..10 or one of "
            f"{sorted(GSO_SCENE_OBJECT_SPECS)}."
        )

    return value


def load_case_config():
    """Load one fixed MuJoCo testcase.

    Normal use selects only the scene number. The corresponding fixed case
    is chosen automatically from DEFAULT_CASE_BY_SCENE.

    File format:
        line 1: natural-language language goal
        line 2: fixed MuJoCo GT target object name

    The language goal is used only by VLM / GroundingDINO / grasp planning.
    The GT target object is independent of VLM output and is used only for
    simulator-side task-success evaluation after grasp execution.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Run one fixed MuJoCo scene. "
            "Use --scene 1..10 to automatically load that scene's fixed case. "
            "Use --case only when an explicit case override is needed."
        )
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="1",
        help=(
            "Scene number 1..10, or scene01..scene10. "
            "Default: 1."
        ),
    )
    parser.add_argument(
        "--case",
        type=str,
        default=None,
        help=(
            "Optional manual case override. "
            "Accepts a case file path or case name under ./cases. "
            "If omitted, the fixed case for --scene is loaded automatically."
        ),
    )
    args = parser.parse_args()

    scene_name = _normalize_scene_name(
        args.scene
    )

    if args.case is None:
        default_case_name = DEFAULT_CASE_BY_SCENE.get(
            scene_name
        )

        if default_case_name is None:
            raise KeyError(
                f"No fixed case is configured for {scene_name!r}."
            )

        requested = CASE_DIR / default_case_name
    else:
        requested = Path(args.case)

        if not requested.is_absolute():
            # A bare case name resolves under ./cases.
            if requested.parent == Path("."):
                if requested.suffix == "":
                    requested = requested.with_suffix(".txt")
                requested = CASE_DIR / requested
            else:
                requested = SCRIPT_DIR / requested

    case_path = requested.resolve()

    if not case_path.is_file():
        if args.case is None:
            raise FileNotFoundError(
                "The fixed case configured for "
                f"{scene_name} does not exist: {case_path}"
            )

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
        "scene_name": scene_name,
    }

# ============================================================
# Perception and Grounding Utilities
# ============================================================

def image_pixel_to_nearest_valid_world_point(
    topview_data,
    pixel_xy,
):
    """Map one heightmap pixel to a valid world XYZ point."""

    pointcloud = np.asarray(
        topview_data["pointcloud"],
        dtype=np.float64,
    )

    height, width = pointcloud.shape[:2]

    pixel_xy = np.asarray(
        pixel_xy,
        dtype=np.float64,
    ).reshape(2)

    px = int(np.clip(round(pixel_xy[0]), 0, width - 1))
    py = int(np.clip(round(pixel_xy[1]), 0, height - 1))

    valid_mask = (
        np.isfinite(pointcloud).all(axis=2)
        & (np.linalg.norm(pointcloud, axis=2) > 1e-9)
    )

    if valid_mask[py, px]:
        return {
            "requested_pixel_xy": np.asarray([px, py], dtype=np.int64),
            "selected_pixel_xy": np.asarray([px, py], dtype=np.int64),
            "world_point": pointcloud[py, px].copy(),
            "fallback_mode": "exact_pixel",
        }

    ys, xs = np.nonzero(valid_mask)
    if len(xs) == 0:
        raise RuntimeError(
            "Heightmap contains no valid world point for VLM centroid mapping."
        )

    d2 = (xs - px) ** 2 + (ys - py) ** 2
    best = int(np.argmin(d2))
    sx = int(xs[best])
    sy = int(ys[best])

    return {
        "requested_pixel_xy": np.asarray([px, py], dtype=np.int64),
        "selected_pixel_xy": np.asarray([sx, sy], dtype=np.int64),
        "world_point": pointcloud[sy, sx].copy(),
        "fallback_mode": "nearest_valid_heightmap_pixel",
    }

def grounding_bbox_to_target_world_xy(
    topview_data,
    bbox_xyxy,
    pixel_margin=GROUNDING_TARGET_GRASP_MARGIN,
):
    """Convert the GroundingDINO target bbox into a world-XY grasp region.

    The original DINO bbox is expanded only by a small target tolerance.
    This is intentionally separate from GROUNDING_CROP_MARGIN:

      target bbox + small margin:
          determines which grasp centres belong to the selected target;

      target bbox + large context margin:
          determines what geometry GraspNet sees.

    No simulator object identity or body centre is used here.
    """

    pointcloud = np.asarray(
        topview_data["pointcloud"],
        dtype=np.float64,
    )

    if (
        pointcloud.ndim != 3
        or pointcloud.shape[2] != 3
    ):
        raise ValueError(
            "Expected topview pointcloud with shape (H, W, 3), "
            f"got {pointcloud.shape}."
        )

    height, width = pointcloud.shape[:2]

    x1, y1, x2, y2 = np.asarray(
        bbox_xyxy,
        dtype=np.float64,
    ).reshape(4)

    margin = int(pixel_margin)

    ix1 = int(
        np.clip(
            np.floor(x1) - margin,
            0,
            width - 1,
        )
    )
    iy1 = int(
        np.clip(
            np.floor(y1) - margin,
            0,
            height - 1,
        )
    )
    ix2 = int(
        np.clip(
            np.ceil(x2) + margin,
            ix1 + 1,
            width,
        )
    )
    iy2 = int(
        np.clip(
            np.ceil(y2) + margin,
            iy1 + 1,
            height,
        )
    )

    region = pointcloud[
        iy1:iy2,
        ix1:ix2,
    ]

    valid_mask = (
        np.isfinite(region).all(axis=2)
        & (
            np.linalg.norm(
                region,
                axis=2,
            )
            > 1e-9
        )
    )

    valid_points = region[
        valid_mask
    ]

    if len(valid_points) == 0:
        raise RuntimeError(
            "GroundingDINO target-grasp region contains "
            "no valid world points."
        )

    xy_min = np.min(
        valid_points[:, :2],
        axis=0,
    )
    xy_max = np.max(
        valid_points[:, :2],
        axis=0,
    )

    return {
        "pixel_bbox_xyxy": np.asarray(
            [ix1, iy1, ix2, iy2],
            dtype=np.int64,
        ),
        "world_xy_min": xy_min,
        "world_xy_max": xy_max,
        "valid_point_count": int(
            len(valid_points)
        ),
    }

# ============================================================
# Grasp Filtering and Selection
# ============================================================

def filter_grasps_to_dino_target_region(
    grasps,
    scores,
    angles_deg,
    target_xy_min,
    target_xy_max,
):
    """Keep grasps whose centres lie inside the DINO target world-XY region."""

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
            (centres_xy[:, 0] >= target_xy_min[0])
            & (centres_xy[:, 0] <= target_xy_max[0])
            & (centres_xy[:, 1] >= target_xy_min[1])
            & (centres_xy[:, 1] <= target_xy_max[1])
        )

        keep_indices = np.flatnonzero(
            keep_mask
        ).astype(np.int64)

    return {
        "grasps": grasps[keep_indices],
        "scores": scores[keep_indices],
        "angles_deg": angles_deg[keep_indices],
        "source_indices": keep_indices,
    }

def select_weighted_grasp(
    grasps,
    scores,
    angles_deg,
    preferred_world_point,
    fallback_mode=False,
):
    """Select one grasp from the complete GroundingDINO target region.

    Weighted target-grasp policy:
      1. Start with every grasp already retained inside the DINO target region.
      2. Convert approach angle to a continuous 0..1 quality score:
             angle_score = exp(-(angle_deg / sigma_angle)^2)
      3. Convert preferred-XY distance to a continuous 0..1 score:
             preferred_score = exp(-(distance_m / sigma_distance)^2)
      4. Compute:
             final_score =
                 0.60 * angle_score
                 + 0.40 * preferred_score
      5. Select the grasp with the largest final_score.

    No hard angle threshold is applied.

    When ``fallback_mode`` is True, the selector is used for full-scene
    decluttering rather than target-grasp selection. In that mode the score is:
        0.60 * angle_score + 0.40 * GraspNet score
    The VLM preferred-location term is intentionally ignored because fallback
    only needs one mechanically friendly grasp that changes the scene.

    In normal target-grasp mode, GraspNet score remains diagnostic only.

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
    ):
        raise ValueError(
            "Grasp / score / angle counts do not match: "
            f"{len(grasps)}, {len(scores)}, {len(angles_deg)}"
        )

    xy_distances = np.linalg.norm(
        grasps[:, :2]
        - preferred_world_point[None, :2],
        axis=1,
    )

    angle_scores = np.exp(
        -(
            angles_deg
            / float(GRASP_SELECTION_ANGLE_SIGMA_DEG)
        ) ** 2
    )

    preferred_scores = np.exp(
        -(
            xy_distances
            / float(GRASP_SELECTION_PREFERRED_SIGMA_M)
        ) ** 2
    )

    if fallback_mode:
        # Full-scene fallback is a simple scene-change action.
        # Favor mechanically friendly approach angles, while still requiring
        # reasonable GraspNet confidence. Target preferred location is ignored.
        final_scores = (
            float(FULL_SCENE_FALLBACK_ANGLE_WEIGHT)
            * angle_scores
            + float(FULL_SCENE_FALLBACK_GRASPNET_WEIGHT)
            * scores
        )
        selection_mode = "full_scene_fallback_angle_plus_graspnet"
        preferred_weight = 0.0
    else:
        # Normal target-grasp selection keeps the VLM-guided policy.
        final_scores = (
            float(GRASP_SELECTION_ANGLE_WEIGHT)
            * angle_scores
            + float(GRASP_SELECTION_PREFERRED_WEIGHT)
            * preferred_scores
        )
        selection_mode = "weighted_angle_preferred"
        preferred_weight = float(GRASP_SELECTION_PREFERRED_WEIGHT)

    selected_index = int(
        np.argmax(final_scores)
    )

    all_indices = np.arange(
        len(grasps),
        dtype=np.int64,
    )

    return {
        "selected_grasp": grasps[selected_index].copy(),
        "selected_index": selected_index,
        "selected_score": float(scores[selected_index]),
        "selected_xy_distance": float(xy_distances[selected_index]),
        "selected_angle_deg": float(angles_deg[selected_index]),

        # Weighted-ranking diagnostics.
        "selected_angle_score": float(
            angle_scores[selected_index]
        ),
        "selected_preferred_score": float(
            preferred_scores[selected_index]
        ),
        "selected_final_score": float(
            final_scores[selected_index]
        ),
        "angle_weight": float(
            FULL_SCENE_FALLBACK_ANGLE_WEIGHT
            if fallback_mode
            else GRASP_SELECTION_ANGLE_WEIGHT
        ),
        "preferred_weight": float(
            preferred_weight
        ),
        "graspnet_weight": float(
            FULL_SCENE_FALLBACK_GRASPNET_WEIGHT
            if fallback_mode
            else 0.0
        ),
        "angle_sigma_deg": float(
            GRASP_SELECTION_ANGLE_SIGMA_DEG
        ),
        "preferred_sigma_m": float(
            GRASP_SELECTION_PREFERRED_SIGMA_M
        ),
        "all_angle_scores": angle_scores.copy(),
        "all_preferred_scores": preferred_scores.copy(),
        "all_final_scores": final_scores.copy(),

        "selection_mode": selection_mode,
        "selection_pool_indices": all_indices.copy(),
        "selection_pool_xy_distances": xy_distances.copy(),
        "selection_pool_scores": scores.copy(),
        "selection_pool_angles_deg": angles_deg.copy(),
    }

def derive_raw_workspace_crop_from_world(
    raw_topview_data,
    workspace_limits,
):
    """Derive the RAW-image crop directly from the configured world workspace.

    No fixed purple/image crop is used.

    We use the RAW top-view per-pixel world pointcloud:
        RAW pixel -> world XYZ

    A pixel belongs to the image-space workspace iff its world XYZ lies inside
    env.perception_workspace_limits. The minimal axis-aligned RAW-image bbox
    covering all such pixels becomes the GroundingDINO input crop.
    """

    pointcloud = np.asarray(
        raw_topview_data["pointcloud"],
        dtype=np.float64,
    )

    workspace_limits = np.asarray(
        workspace_limits,
        dtype=np.float64,
    )

    if pointcloud.ndim != 3 or pointcloud.shape[2] != 3:
        raise ValueError(
            "Expected raw top-view pointcloud with shape (H, W, 3), "
            f"got {pointcloud.shape}."
        )

    if workspace_limits.shape != (3, 2):
        raise ValueError(
            "workspace_limits must have shape (3, 2), "
            f"got {workspace_limits.shape}."
        )

    valid = (
        np.isfinite(pointcloud).all(axis=2)
        & (np.linalg.norm(pointcloud, axis=2) > 1e-9)
        & (pointcloud[..., 0] >= workspace_limits[0, 0])
        & (pointcloud[..., 0] <= workspace_limits[0, 1])
        & (pointcloud[..., 1] >= workspace_limits[1, 0])
        & (pointcloud[..., 1] <= workspace_limits[1, 1])
        & (pointcloud[..., 2] >= workspace_limits[2, 0])
        & (pointcloud[..., 2] <= workspace_limits[2, 1])
    )

    ys, xs = np.nonzero(valid)

    if len(xs) == 0:
        raise RuntimeError(
            "No RAW top-view pixels project into the configured world workspace."
        )

    height, width = pointcloud.shape[:2]

    x1 = int(np.clip(xs.min(), 0, width - 1))
    y1 = int(np.clip(ys.min(), 0, height - 1))
    x2 = int(np.clip(xs.max() + 1, x1 + 1, width))
    y2 = int(np.clip(ys.max() + 1, y1 + 1, height))

    return {
        "crop_xyxy": np.asarray(
            [x1, y1, x2, y2],
            dtype=np.int64,
        ),
        "workspace_pixel_count": int(len(xs)),
        "mask": valid,
    }

def raw_crop_bbox_to_full_bbox(
    bbox_xyxy,
    crop_xyxy,
):
    """Translate one GroundingDINO bbox from crop coordinates to full RAW coordinates."""

    bbox = np.asarray(
        bbox_xyxy,
        dtype=np.float64,
    ).reshape(4)

    crop = np.asarray(
        crop_xyxy,
        dtype=np.float64,
    ).reshape(4)

    offset_x = crop[0]
    offset_y = crop[1]

    return np.asarray(
        [
            bbox[0] + offset_x,
            bbox[1] + offset_y,
            bbox[2] + offset_x,
            bbox[3] + offset_y,
        ],
        dtype=np.float64,
    )

def raw_bbox_to_workspace_world_region(
    raw_topview_data,
    bbox_xyxy,
    workspace_limits,
):
    """Convert one full-RAW DINO bbox to a world-XYZ region.

    Only valid points inside the configured world workspace are retained.
    """

    pointcloud = np.asarray(
        raw_topview_data["pointcloud"],
        dtype=np.float64,
    )

    workspace_limits = np.asarray(
        workspace_limits,
        dtype=np.float64,
    )

    height, width = pointcloud.shape[:2]

    x1, y1, x2, y2 = np.asarray(
        bbox_xyxy,
        dtype=np.float64,
    ).reshape(4)

    ix1 = int(np.clip(np.floor(x1), 0, width - 1))
    iy1 = int(np.clip(np.floor(y1), 0, height - 1))
    ix2 = int(np.clip(np.ceil(x2), ix1 + 1, width))
    iy2 = int(np.clip(np.ceil(y2), iy1 + 1, height))

    region = pointcloud[iy1:iy2, ix1:ix2]

    valid = (
        np.isfinite(region).all(axis=2)
        & (np.linalg.norm(region, axis=2) > 1e-9)
        & (region[..., 0] >= workspace_limits[0, 0])
        & (region[..., 0] <= workspace_limits[0, 1])
        & (region[..., 1] >= workspace_limits[1, 0])
        & (region[..., 1] <= workspace_limits[1, 1])
        & (region[..., 2] >= workspace_limits[2, 0])
        & (region[..., 2] <= workspace_limits[2, 1])
    )

    valid_points = region[valid]

    if len(valid_points) == 0:
        raise RuntimeError(
            "RAW GroundingDINO bbox contains no valid 3D points inside "
            "the official world workspace."
        )

    return {
        "pixel_bbox_xyxy": np.asarray(
            [ix1, iy1, ix2, iy2],
            dtype=np.int64,
        ),
        "world_xyz_min": np.min(valid_points, axis=0),
        "world_xyz_max": np.max(valid_points, axis=0),
        "world_xy_min": np.min(valid_points[:, :2], axis=0),
        "world_xy_max": np.max(valid_points[:, :2], axis=0),
        "valid_point_count": int(len(valid_points)),
    }

def world_xy_region_to_heightmap_bbox(
    topview_data,
    world_xy_min,
    world_xy_max,
):
    """Register a world-XY rectangle onto the existing heightmap grid."""

    pointcloud = np.asarray(
        topview_data["pointcloud"],
        dtype=np.float64,
    )

    world_xy_min = np.asarray(
        world_xy_min,
        dtype=np.float64,
    ).reshape(2)

    world_xy_max = np.asarray(
        world_xy_max,
        dtype=np.float64,
    ).reshape(2)

    valid = (
        np.isfinite(pointcloud).all(axis=2)
        & (np.linalg.norm(pointcloud, axis=2) > 1e-9)
        & (pointcloud[..., 0] >= world_xy_min[0])
        & (pointcloud[..., 0] <= world_xy_max[0])
        & (pointcloud[..., 1] >= world_xy_min[1])
        & (pointcloud[..., 1] <= world_xy_max[1])
    )

    ys, xs = np.nonzero(valid)

    if len(xs) == 0:
        raise RuntimeError(
            "RAW DINO world-XY region does not overlap valid heightmap pixels."
        )

    height, width = pointcloud.shape[:2]

    x1 = int(np.clip(xs.min(), 0, width - 1))
    y1 = int(np.clip(ys.min(), 0, height - 1))
    x2 = int(np.clip(xs.max() + 1, x1 + 1, width))
    y2 = int(np.clip(ys.max() + 1, y1 + 1, height))

    return {
        "bbox_xyxy": np.asarray(
            [x1, y1, x2, y2],
            dtype=np.float64,
        ),
        "matching_heightmap_point_count": int(len(xs)),
    }

def preferred_location_to_world_point_raw(
    raw_topview_data,
    bbox_xyxy,
    preferred_location,
    workspace_limits,
):
    """Map the preferred 3x3 cell of a full-RAW DINO bbox to world XYZ."""

    preferred_location = int(preferred_location)
    if not 1 <= preferred_location <= 9:
        preferred_location = 5

    pointcloud = np.asarray(
        raw_topview_data["pointcloud"],
        dtype=np.float64,
    )

    workspace_limits = np.asarray(
        workspace_limits,
        dtype=np.float64,
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

    row = (preferred_location - 1) // 3
    col = (preferred_location - 1) % 3

    cell_x1 = x1 + col * (x2 - x1) / 3.0
    cell_x2 = x1 + (col + 1) * (x2 - x1) / 3.0
    cell_y1 = y1 + row * (y2 - y1) / 3.0
    cell_y2 = y1 + (row + 1) * (y2 - y1) / 3.0

    req_x = int(np.clip(round((cell_x1 + cell_x2) / 2.0), 0, width - 1))
    req_y = int(np.clip(round((cell_y1 + cell_y2) / 2.0), 0, height - 1))

    def valid_point(p):
        p = np.asarray(p, dtype=np.float64).reshape(3)
        return bool(
            np.isfinite(p).all()
            and np.linalg.norm(p) > 1e-9
            and workspace_limits[0, 0] <= p[0] <= workspace_limits[0, 1]
            and workspace_limits[1, 0] <= p[1] <= workspace_limits[1, 1]
            and workspace_limits[2, 0] <= p[2] <= workspace_limits[2, 1]
        )

    requested_world = pointcloud[req_y, req_x].copy()

    if valid_point(requested_world):
        return {
            "preferred_location": preferred_location,
            "cell_xyxy": np.asarray(
                [cell_x1, cell_y1, cell_x2, cell_y2],
                dtype=np.float64,
            ),
            "requested_center_pixel_xy": np.asarray(
                [req_x, req_y],
                dtype=np.int64,
            ),
            "center_pixel_xy": np.asarray(
                [req_x, req_y],
                dtype=np.int64,
            ),
            "world_point": requested_world,
            "fallback_mode": "raw_cell_center",
        }

    def nearest_valid(search_xyxy):
        sx1, sy1, sx2, sy2 = np.asarray(
            search_xyxy,
            dtype=np.float64,
        ).reshape(4)

        ix1 = int(np.clip(np.floor(sx1), 0, width - 1))
        iy1 = int(np.clip(np.floor(sy1), 0, height - 1))
        ix2 = int(np.clip(np.ceil(sx2), ix1 + 1, width))
        iy2 = int(np.clip(np.ceil(sy2), iy1 + 1, height))

        region = pointcloud[iy1:iy2, ix1:ix2]

        valid = (
            np.isfinite(region).all(axis=2)
            & (np.linalg.norm(region, axis=2) > 1e-9)
            & (region[..., 0] >= workspace_limits[0, 0])
            & (region[..., 0] <= workspace_limits[0, 1])
            & (region[..., 1] >= workspace_limits[1, 0])
            & (region[..., 1] <= workspace_limits[1, 1])
            & (region[..., 2] >= workspace_limits[2, 0])
            & (region[..., 2] <= workspace_limits[2, 1])
        )

        ys, xs = np.nonzero(valid)
        if len(xs) == 0:
            return None

        gx = xs + ix1
        gy = ys + iy1

        d2 = (gx - req_x) ** 2 + (gy - req_y) ** 2
        best = int(np.argmin(d2))

        px = int(gx[best])
        py = int(gy[best])

        return (
            np.asarray([px, py], dtype=np.int64),
            pointcloud[py, px].copy(),
        )

    selected = nearest_valid([cell_x1, cell_y1, cell_x2, cell_y2])
    fallback_mode = "nearest_valid_in_raw_preferred_cell"

    if selected is None:
        selected = nearest_valid([x1, y1, x2, y2])
        fallback_mode = "nearest_valid_in_raw_dino_bbox"

    if selected is None:
        raise RuntimeError(
            "No valid workspace point exists in preferred RAW DINO region."
        )

    selected_pixel, selected_world = selected

    return {
        "preferred_location": preferred_location,
        "cell_xyxy": np.asarray(
            [cell_x1, cell_y1, cell_x2, cell_y2],
            dtype=np.float64,
        ),
        "requested_center_pixel_xy": np.asarray(
            [req_x, req_y],
            dtype=np.int64,
        ),
        "center_pixel_xy": selected_pixel,
        "world_point": selected_world,
        "fallback_mode": fallback_mode,
    }

# ============================================================
# Full-Scene Fallback and Point-Cloud Fusion
# ============================================================

def run_pointcloud_fusion(
    raw_views_path,
    output_path,
):
    """Run the project-local multi-view point-cloud fusion routine.

    The MuJoCo process supplies four world-frame, workspace-filtered camera
    clouds. The subprocess then applies the reconstruction procedure used by
    the project-local fusion implementation:
        statistical outlier removal
        -> normal estimation
        -> 1.5 mm voxel downsampling
        -> point-to-plane ICP
        -> merge
        -> voxel downsampling / normal re-estimation

    The same reconstruction strategy is used consistently throughout the
    current pipeline.
    """

    command = [
        str(GROUNDING_PYTHON),
        str(POINTCLOUD_FUSION_SCRIPT),
        "--input",
        str(raw_views_path),
        "--output",
        str(output_path),
    ]

    print("Running point-cloud fusion.")

    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
    )

    if not Path(output_path).is_file():
        raise RuntimeError(
            "Point-cloud fusion finished without creating "
            f"the expected output: {output_path}"
        )

def _derive_full_scene_workspace_xy_bounds(env):
    """Return XY bounds from the runtime perception workspace."""

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

    return xy_min, xy_max

def export_workspace_filtered_full_scene(
    env,
    output_path,
    ply_path,
):
    """Export the four-view full scene using the project-local fusion pipeline.

    MuJoCo-specific part:
        capture the four current MuJoCo cameras
        -> use their already world-frame point clouds
        -> keep the validated official MuJoCo workspace
        -> keep the existing tabletop clearance

    Fusion stage:
        each camera cloud is sorted by Z before fusion
        -> subprocess calls the project-local fusion implementation
           with the configured reconstruction parameters.
    """

    workspace_xy_min, workspace_xy_max = (
        _derive_full_scene_workspace_xy_bounds(
            env=env,
        )
    )

    runtime_workspace = np.asarray(
        env.perception_workspace_limits,
        dtype=np.float64,
    )

    # Apply the runtime workspace bounds before multi-view point-cloud fusion.
    z_min = max(
        float(runtime_workspace[2, 0]),
        float(FULL_SCENE_TABLE_HEIGHT_M)
        + float(FULL_SCENE_TABLE_CLEARANCE_M),
    )
    z_max = min(
        float(runtime_workspace[2, 1]),
        float(FULL_SCENE_TABLE_HEIGHT_M)
        + float(FULL_SCENE_MAX_HEIGHT_ABOVE_TABLE_M),
    )

    if z_min >= z_max:
        raise RuntimeError(
            "Workspace Z range is empty after tabletop clearance."
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

        # Use lower-inclusive / upper-exclusive workspace bounds.
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
                "Full-scene fusion received an empty "
                f"workspace cloud from camera {camera_name!r}."
            )

        # Sort each camera cloud by Z before fusion.
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

    run_pointcloud_fusion(
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

        fusion_voxel_size = float(
            fused_data["voxel_size"]
        )
        fusion_nb_neighbors = int(
            fused_data["nb_neighbors"]
        )
        fusion_std_ratio = float(
            fused_data["std_ratio"]
        )
        fusion_max_correspondence_distance = float(
            fused_data["max_correspondence_distance"]
        )

    export_colored_pointcloud_ply(
        output_path=ply_path,
        points=points,
        colors=colors,
    )

    print(
        "Full-scene fusion:",
        f"{len(points)} points from {len(FULL_SCENE_CAMERAS)} views, "
        f"voxel={fusion_voxel_size:.4f} m",
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
        "fusion_voxel_size": fusion_voxel_size,
    }

def run_groundingdino_detection(
    image,
    text_prompt,
    visualization_path,
):
    """Run GroundingDINO in the configured perception environment."""

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
        str(visualization_path),
        "--box-threshold",
        str(GROUNDING_BOX_THRESHOLD),
        "--text-threshold",
        str(GROUNDING_TEXT_THRESHOLD),
        "--device",
        "cuda",
    ]

    print("Running GroundingDINO.")

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

# ============================================================
# Visualization and Diagnostics
# ============================================================

def save_combined_grounding_visualization(
    image,
    boxes_xyxy,
    final_bbox_scores,
    selected_index,
    preferred_location,
    preferred_pixel_xy,
    output_path,
):
    """Save one per-attempt GroundingDINO overview image.

    The image contains:
      - every GroundingDINO candidate bbox with only its final weighted score;
      - the selected bbox highlighted with a thick blue border + SELECTED;
      - the selected bbox split into the 3x3 preferred grasp grid;
      - the preferred cell highlighted;
      - the preferred world-point source pixel marked.

    All bbox coordinates must be in the same image coordinate system as
    ``image``.
    """

    image = np.asarray(image)

    if image.dtype != np.uint8:
        clipped = np.clip(image, 0, 255)
        image = clipped.astype(np.uint8)

    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)

    try:
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                20,
            )
        except Exception:
            font = ImageFont.load_default()
    except Exception:
        font = None

    boxes = np.asarray(
        boxes_xyxy,
        dtype=np.float64,
    ).reshape(-1, 4)
    selected_index = int(selected_index)
    preferred_location = int(preferred_location)

    # 1) Draw all candidates first.
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = [float(v) for v in box]
        is_selected = (idx == selected_index)

        if is_selected:
            outline = (0, 120, 255)  # blue
            width = 5
        else:
            outline = (255, 0, 0)    # red
            width = 2

        draw.rectangle(
            [x1, y1, x2, y2],
            outline=outline,
            width=width,
        )

        # Display only the score that actually controls bbox selection.
        # final_bbox_score = 0.70 * DINO + 0.30 * centroid proximity.
        label = f"{float(final_bbox_scores[idx]):.3f}"

        # Keep label inside image as much as possible.
        label_x = max(0, int(round(x1)))
        label_y = max(0, int(round(y1)) - 13)

        draw.text(
            (label_x, label_y),
            label,
            fill=outline,
            font=font,
            stroke_width=2,
            stroke_fill=(255, 255, 255),
        )

    # 2) Overlay 3x3 grid only on selected bbox.
    x1, y1, x2, y2 = [
        float(v)
        for v in boxes[selected_index]
    ]

    cell_w = (x2 - x1) / 3.0
    cell_h = (y2 - y1) / 3.0

    grid_color = (0, 220, 255)
    for split in (1, 2):
        gx = x1 + split * cell_w
        gy = y1 + split * cell_h

        draw.line(
            [(gx, y1), (gx, y2)],
            fill=grid_color,
            width=2,
        )
        draw.line(
            [(x1, gy), (x2, gy)],
            fill=grid_color,
            width=2,
        )

    # Number cells 1..9.
    for cell in range(1, 10):
        row = (cell - 1) // 3
        col = (cell - 1) % 3

        cx1 = x1 + col * cell_w
        cy1 = y1 + row * cell_h
        cx2 = x1 + (col + 1) * cell_w
        cy2 = y1 + (row + 1) * cell_h

        center_x = 0.5 * (cx1 + cx2)
        center_y = 0.5 * (cy1 + cy2)

        if cell == preferred_location:
            draw.rectangle(
                [cx1, cy1, cx2, cy2],
                outline=(255, 215, 0),
                width=4,
            )

        # Small dark badge for the cell number.
        r = 10
        draw.ellipse(
            [
                center_x - r,
                center_y - r,
                center_x + r,
                center_y + r,
            ],
            fill=(55, 55, 55),
        )
        try:
            num_bbox = draw.textbbox(
                (0, 0),
                str(cell),
                font=font,
            )
            num_w = num_bbox[2] - num_bbox[0]
            num_h = num_bbox[3] - num_bbox[1]
        except Exception:
            num_w, num_h = 8, 12

        draw.text(
            (center_x - num_w / 2.0, center_y - num_h / 2.0),
            str(cell),
            fill=(255, 255, 255),
            font=font,
        )

    # 3) Mark preferred source pixel.
    preferred_pixel_xy = np.asarray(
        preferred_pixel_xy,
        dtype=np.float64,
    ).reshape(2)

    px, py = [
        float(v)
        for v in preferred_pixel_xy
    ]

    r = 8
    draw.ellipse(
        [px - r, py - r, px + r, py + r],
        fill=(255, 80, 80),
        outline=(255, 255, 255),
        width=3,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    pil_image.save(output_path)

    return output_path

def _build_grasp_debug_marker_points(
    env,
    grasp,
):
    """Return world-frame marker points for one 7D grasp.

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

    # End-effector convention used throughout this runner:
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
        COMPAT_WORKSPACE_PREVIEW_DIR,
        COMPAT_VIEW_TESTS_DIR,
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

# ============================================================
# Closed-Loop Task Execution
# ============================================================

def main():
    case_config = load_case_config()
    vlm_goal = case_config["language_goal"]
    target_object = case_config["target_object"]
    scene_name = case_config["scene_name"]

    clear_previous_closed_loop_outputs()

    print("Scene:", scene_name)
    print("Case file:", case_config["case_path"])
    print("Language goal:", vlm_goal)
    print("Simulator GT target for final task evaluation only:", target_object)

    controller_config = (
        load_thinkgrasp_joint_position_controller_config()
    )

    print("Grasp control mode: IK + q_ref + JOINT_POSITION")

    env = ThinkGraspMinimalEnv(
        controller_configs=controller_config,
        hard_reset=False,
        scene_name=scene_name,
    )

    print(
        "Scene objects:",
        sorted(env.object_body_ids.keys()),
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
        # Environment construction already performs the initial reset
        # and clutter generation. Do not reset again here, otherwise a second
        # valid clutter scene would be generated and replace the first one.

        # Save the initial seven Panda joint positions as the home
        # configuration. Failure recovery returns here before re-perception.
        home_joint_positions = (
            env.get_arm_joint_positions().copy()
        )

        print(
            "Saved home joint positions:",
            home_joint_positions,
        )
        print(
            "Perception workspace:",
            env.perception_workspace_limits,
        )

        recorder.start()
        recorder.add_hold(1.0)

        env.open_gripper(steps=30)

        target_completed = False
        attempts_started = 0
        ended_no_grasp_after_fallback = False

        episode_reward = 0.0

        max_pos_dist = float(
            np.sqrt(
                (
                    env.perception_workspace_limits[0, 1]
                    - env.perception_workspace_limits[0, 0]
                ) ** 2
                + (
                    env.perception_workspace_limits[1, 1]
                    - env.perception_workspace_limits[1, 0]
                ) ** 2
            )
        )

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

            # ------------------------------------------------------------
            # 1. Perception
            # ------------------------------------------------------------

            # GroundingDINO perception path:
            #   raw top-view RGB-D -> derive RAW crop from the runtime world workspace
            #   -> crop RAW RGB without resizing -> GroundingDINO
            #   -> crop bbox + offset -> full RAW bbox
            #   -> raw per-pixel world XYZ -> official world workspace
            #   -> registered heightmap bbox -> existing GraspNet geometry
            #
            # VLM uses the orthographic workspace heightmap.
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

            # Human-readable diagnostic only. Keep visualization outputs
            # outside bridge_data so bridge_data contains only files that are
            # actually exchanged between pipeline stages.
            PERCEPTION_VIEW_OUTPUT_DIR.mkdir(
                parents=True,
                exist_ok=True,
            )
            imageio.imwrite(
                PERCEPTION_VIEW_OUTPUT_DIR / "raw_topview_rgb.png",
                np.asarray(raw_topview_data["color"]),
            )

            print(
                "Perception:",
                f"RAW={raw_topview_data['color'].shape}, "
                f"heightmap={topview_data['color'].shape}, "
                f"valid_points={topview_data['valid_workspace_point_count']}",
            )

            raw_workspace_crop = derive_raw_workspace_crop_from_world(
                raw_topview_data=raw_topview_data,
                workspace_limits=env.perception_workspace_limits,
            )

            raw_workspace_crop_xyxy = raw_workspace_crop["crop_xyxy"]
            rx1, ry1, rx2, ry2 = raw_workspace_crop_xyxy

            raw_workspace_rgb = np.asarray(
                raw_topview_data["color"]
            )[ry1:ry2, rx1:rx2].copy()

            imageio.imwrite(
                PERCEPTION_VIEW_OUTPUT_DIR / "raw_workspace_rgb.png",
                raw_workspace_rgb,
            )

            print(
                "RAW workspace crop:",
                f"bbox={raw_workspace_crop_xyxy.tolist()}, shape={raw_workspace_rgb.shape}",
            )

            # ------------------------------------------------------------
            # 2. VLM Target Selection
            # ------------------------------------------------------------

            # VLM selection:
            #   external natural-language goal + same top-view RGB
            #   -> selected object + preferred 3x3 grasp location

            print()
            print("Running Qwen3-VL selection.")

            vlm_result = run_vlm_selection(
                image_path=FULL_PERCEPTION_IMAGE_PATH,
                goal=vlm_goal,
                system_prompt_path=VLM_SYSTEM_PROMPT_PATH,
            )

            vlm_selected_object = str(
                vlm_result["selected_object"]
            ).strip()

            vlm_selection_reason = (
                vlm_result.get("selection_reason")
                or "not provided"
            )

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

            print(
                "VLM:",
                f"selected={vlm_selected_object!r}, reason={vlm_selection_reason!r}, "
                f"preferred={preferred_grasping_location}",
            )

            vlm_centroid_world_result = (
                image_pixel_to_nearest_valid_world_point(
                    topview_data=topview_data,
                    pixel_xy=vlm_selected_centroid_xy,
                )
            )
            vlm_centroid_world_point = np.asarray(
                vlm_centroid_world_result["world_point"],
                dtype=np.float64,
            ).reshape(3)

            print(
                "VLM centroid:",
                f"pixel={vlm_centroid_world_result['selected_pixel_xy'].tolist()}, "
                f"world_xy={np.round(vlm_centroid_world_point[:2], 4).tolist()}, "
                f"mapping={vlm_centroid_world_result['fallback_mode']}",
            )

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
                centroid_xy=vlm_selected_centroid_xy,
                output_path=vlm_viz_path,
            )

            # ------------------------------------------------------------
            # 3. GroundingDINO Target Localization
            # ------------------------------------------------------------

            # GroundingDINO receives the object name selected by Qwen.
            # The internal MuJoCo target name is resolved separately above.
            grounding_prompt = (
                vlm_selected_object
                .replace("_", " ")
            )

            print("GroundingDINO prompt:", grounding_prompt)

            GROUNDING_GRID_OUTPUT_DIR.mkdir(
                parents=True,
                exist_ok=True,
            )
            grounding_grid_viz_path = (
                GROUNDING_GRID_OUTPUT_DIR
                / (
                    f"{target_object}_attempt_{attempt:02d}_"
                    "grounding_grid.png"
                )
            )

            # GroundingDINO's subprocess visualization is written directly to
            # the per-attempt diagnostic output path, never to bridge_data.
            # After candidate selection, this same path is overwritten by the
            # richer combined candidate + selected + 3x3 visualization.
            grounding_result = run_groundingdino_detection(
                image=raw_workspace_rgb,
                text_prompt=grounding_prompt,
                visualization_path=grounding_grid_viz_path,
            )

            raw_boxes = grounding_result["boxes"]
            raw_scores = grounding_result["scores"]
            raw_phrases = grounding_result["phrases"]

            # DINO bbox coordinates are relative to raw_workspace_rgb.
            # Convert every candidate back to full 640x640 RAW coordinates.
            if len(raw_boxes) > 0:
                full_raw_boxes = np.stack(
                    [
                        raw_crop_bbox_to_full_bbox(
                            bbox_xyxy=box,
                            crop_xyxy=raw_workspace_crop_xyxy,
                        )
                        for box in raw_boxes
                    ],
                    axis=0,
                )
            else:
                full_raw_boxes = np.empty(
                    (0, 4),
                    dtype=np.float64,
                )

            boxes = full_raw_boxes
            scores = np.asarray(
                raw_scores,
                dtype=np.float64,
            )
            phrases = np.asarray(
                raw_phrases,
            )

            print(
                "GroundingDINO candidates:",
                len(boxes),
            )

            if len(boxes) == 0:
                print(
                    "GroundingDINO found no target bbox inside the "
                    "perception workspace. Re-perceiving the scene."
                )
                continue

            # GroundingDINO bbox soft ranking:
            # DINO confidence is primary evidence; VLM centroid proximity
            # is auxiliary evidence. Centroid distance is measured in world XY.
            dino_centroid_distances_m = np.full(
                len(boxes),
                np.inf,
                dtype=np.float64,
            )
            dino_centroid_scores = np.zeros(
                len(boxes),
                dtype=np.float64,
            )
            dino_final_bbox_scores = np.full(
                len(boxes),
                -np.inf,
                dtype=np.float64,
            )
            dino_candidate_world_xy_centres = np.full(
                (len(boxes), 2),
                np.nan,
                dtype=np.float64,
            )

            for bbox_index, full_box in enumerate(boxes):
                try:
                    bbox_world_region = (
                        raw_bbox_to_workspace_world_region(
                            raw_topview_data=raw_topview_data,
                            bbox_xyxy=full_box,
                            workspace_limits=env.perception_workspace_limits,
                        )
                    )

                    bbox_world_xy_center = 0.5 * (
                        np.asarray(
                            bbox_world_region["world_xy_min"],
                            dtype=np.float64,
                        )
                        + np.asarray(
                            bbox_world_region["world_xy_max"],
                            dtype=np.float64,
                        )
                    )

                    centroid_distance_m = float(
                        np.linalg.norm(
                            bbox_world_xy_center
                            - vlm_centroid_world_point[:2]
                        )
                    )

                    centroid_score = float(
                        np.exp(
                            -(
                                centroid_distance_m
                                / float(DINO_RANKING_CENTROID_SIGMA_M)
                            ) ** 2
                        )
                    )

                    final_bbox_score = (
                        float(DINO_RANKING_CONFIDENCE_WEIGHT)
                        * float(scores[bbox_index])
                        + float(DINO_RANKING_CENTROID_WEIGHT)
                        * centroid_score
                    )

                    dino_candidate_world_xy_centres[bbox_index] = (
                        bbox_world_xy_center
                    )
                    dino_centroid_distances_m[bbox_index] = centroid_distance_m
                    dino_centroid_scores[bbox_index] = centroid_score
                    dino_final_bbox_scores[bbox_index] = final_bbox_score
                except RuntimeError:
                    pass

            candidate_order = np.argsort(
                -dino_final_bbox_scores,
            )

            print(
                "DINO ranking "
                f"(confidence={DINO_RANKING_CONFIDENCE_WEIGHT:.2f}, "
                f"centroid={DINO_RANKING_CENTROID_WEIGHT:.2f}):"
            )

            for rank, ranked_index in enumerate(
                candidate_order,
                start=1,
            ):
                ranked_index = int(ranked_index)

                full_box = boxes[ranked_index]
                full_center = np.asarray(
                    [
                        0.5 * (full_box[0] + full_box[2]),
                        0.5 * (full_box[1] + full_box[3]),
                    ],
                    dtype=np.float64,
                )

                print(
                    f"  rank {rank}: "
                    f"index={ranked_index}, "
                    f"phrase={str(phrases[ranked_index]).strip()!r}, "
                    f"full_raw_center={np.round(full_center, 2)}, "
                    f"dino_score={float(scores[ranked_index]):.4f}, "
                    f"bbox_world_xy={np.round(dino_candidate_world_xy_centres[ranked_index], 4)}, "
                    f"centroid_distance_m={float(dino_centroid_distances_m[ranked_index]):.4f}, "
                    f"centroid_score={float(dino_centroid_scores[ranked_index]):.4f}, "
                    f"final_bbox_score={float(dino_final_bbox_scores[ranked_index]):.4f}"
                )

            selected_candidate_index = None
            selected_scene_result = None
            selected_heightmap_box = None

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

                print(
                    f"Checking DINO candidate {candidate_rank}/{len(candidate_order)}: "
                    f"phrase={candidate_phrase!r}, score={candidate_score:.4f}"
                )

                try:
                    raw_world_region = raw_bbox_to_workspace_world_region(
                        raw_topview_data=raw_topview_data,
                        bbox_xyxy=candidate_box,
                        workspace_limits=env.perception_workspace_limits,
                    )

                    registered_heightmap = world_xy_region_to_heightmap_bbox(
                        topview_data=topview_data,
                        world_xy_min=raw_world_region["world_xy_min"],
                        world_xy_max=raw_world_region["world_xy_max"],
                    )

                    candidate_heightmap_box = (
                        registered_heightmap["bbox_xyxy"]
                    )

                    candidate_scene_result = (
                        export_pybullet_style_target_scene(
                            output_path=SCENE_PATH,
                            heightmap_data=topview_data,
                            bbox_xyxy=candidate_heightmap_box,
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
                selected_heightmap_box = (
                    candidate_heightmap_box
                )

                print(
                    "Candidate accepted:",
                    f"workspace_points={target_point_count}, "
                    f"positive_pixels={candidate_scene_result['positive_bbox_pixel_count']}",
                )
                break

            if selected_candidate_index is None:
                print(
                    "No GroundingDINO candidate produced a valid "
                    "target crop. Re-perceiving."
                )
                continue

            scene_result = selected_scene_result
            selected_box = boxes[selected_candidate_index]

            # -----------------------------------------------------
            # Two-layer GroundingDINO region policy.
            #
            # Outer region:
            #   selected_box + GROUNDING_CROP_MARGIN
            #   -> GraspNet context / collision geometry.
            #
            # Inner region:
            #   selected_box + GROUNDING_TARGET_GRASP_MARGIN
            #   -> final grasp-centre eligibility.
            # -----------------------------------------------------
            target_grasp_region = (
                grounding_bbox_to_target_world_xy(
                    topview_data=topview_data,
                    bbox_xyxy=selected_heightmap_box,
                    pixel_margin=(
                        GROUNDING_TARGET_GRASP_MARGIN
                    ),
                )
            )

            print(
                "Selected DINO candidate:",
                f"index={selected_candidate_index}, "
                f"phrase={str(phrases[selected_candidate_index]).strip()!r}, "
                f"dino={float(scores[selected_candidate_index]):.4f}, "
                f"centroid_dist={float(dino_centroid_distances_m[selected_candidate_index]):.4f} m, "
                f"final={float(dino_final_bbox_scores[selected_candidate_index]):.4f}",
            )
            print(
                "Target region:",
                f"{target_grasp_region['valid_point_count']} valid points",
            )

            preferred_point_result = (
                preferred_location_to_world_point_raw(
                    raw_topview_data=raw_topview_data,
                    bbox_xyxy=selected_box,
                    preferred_location=(
                        preferred_grasping_location
                    ),
                    workspace_limits=(
                        env.perception_workspace_limits
                    ),
                )
            )

            preferred_world_point = (
                preferred_point_result[
                    "world_point"
                ]
            )

            print(
                "Preferred grasp point:",
                f"cell={preferred_point_result['preferred_location']}, "
                f"world_xy={np.round(preferred_world_point[:2], 4).tolist()}, "
                f"mapping={preferred_point_result['fallback_mode']}",
            )

            preferred_pixel_xy_crop = (
                np.asarray(
                    preferred_point_result[
                        "center_pixel_xy"
                    ],
                    dtype=np.int64,
                )
                - np.asarray(
                    raw_workspace_crop_xyxy[:2],
                    dtype=np.int64,
                )
            )

            save_combined_grounding_visualization(
                image=raw_workspace_rgb,
                boxes_xyxy=raw_boxes,
                final_bbox_scores=dino_final_bbox_scores,
                selected_index=selected_candidate_index,
                preferred_location=preferred_grasping_location,
                preferred_pixel_xy=preferred_pixel_xy_crop,
                output_path=grounding_grid_viz_path,
            )

            # Diagnostic RGB snapshots. The saved topview image is the
            # orthographic heightmap used by VLM and target-crop planning;
            # GroundingDINO uses the RAW workspace crop.
            PERCEPTION_VIEW_OUTPUT_DIR.mkdir(
                parents=True,
                exist_ok=True,
            )

            perception_view_configs = (
                ("topview", 640, 640),
                ("front_oblique_25deg", 640, 480),
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

                # Keep the existing flip only for the custom left oblique
                # camera, whose saved image convention remains vertically
                # inverted relative to the other diagnostic views.
                if diagnostic_camera_name in (
                    "left_oblique_25deg",
                ):
                    diagnostic_image = np.flipud(
                        diagnostic_image
                    ).copy()

                imageio.imwrite(
                    diagnostic_image_path,
                    diagnostic_image,
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

            # ------------------------------------------------------------
            # 4. Grasp Generation and Selection
            # ------------------------------------------------------------

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

            # Keep the expanded DINO crop as GraspNet context, but only
            # allow grasp centres inside the smaller DINO target region to
            # participate in final selection.
            target_region_filtered = (
                filter_grasps_to_dino_target_region(
                    grasps=grasp_data["grasps"],
                    scores=grasp_data["scores"],
                    angles_deg=grasp_data["angles_deg"],
                    target_xy_min=(
                        target_grasp_region[
                            "world_xy_min"
                        ]
                    ),
                    target_xy_max=(
                        target_grasp_region[
                            "world_xy_max"
                        ]
                    ),
                )
            )

            print(
                "GraspNet:",
                f"collision_filtered={len(grasp_data['grasps'])}, "
                f"target_region={len(target_region_filtered['grasps'])}",
            )

            # The DINO target region defines the final target-grasp
            # candidate pool. Nearest-object assignment is not used.
            candidate_grasps = np.asarray(
                target_region_filtered["grasps"],
                dtype=np.float64,
            ).reshape(-1, 7)
            candidate_scores = np.asarray(
                target_region_filtered["scores"],
                dtype=np.float64,
            ).reshape(-1)
            candidate_angles = np.asarray(
                target_region_filtered["angles_deg"],
                dtype=np.float64,
            ).reshape(-1)

            used_full_scene_fallback = False

            if len(candidate_grasps) == 0:
                print()
                print(
                    "Target region produced 0 usable grasps. "
                    "Running full-scene fallback."
                )

                # Diagnostic archive only:
                # preserve the exact full-scene fused point cloud used by
                # this fallback attempt. This does not recompute or modify
                # the fallback point cloud.
                fallback_fused_cloud_ply_path = (
                    FUSED_CLOUD_PREVIEW_DIR
                    / (
                        f"{target_object}_attempt_{attempt}_"
                        "full_scene_fallback.ply"
                    )
                )

                fallback_exported_ply_path = (
                    export_colored_pointcloud_ply(
                        output_path=fallback_fused_cloud_ply_path,
                        points=full_scene_debug_points,
                        colors=full_scene_debug_colors,
                    )
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

                candidate_grasps = np.asarray(
                    full_grasp_data["grasps"],
                    dtype=np.float64,
                ).reshape(-1, 7)
                candidate_scores = np.asarray(
                    full_grasp_data["scores"],
                    dtype=np.float64,
                ).reshape(-1)
                candidate_angles = np.asarray(
                    full_grasp_data["angles_deg"],
                    dtype=np.float64,
                ).reshape(-1)

                print(
                    "Usable full-scene fallback grasps:",
                    len(candidate_grasps),
                )

                if len(candidate_grasps) == 0:
                    print(
                        "Full-scene fallback produced no usable grasp."
                    )
                    print(
                        "Ending this case instead of re-perceiving an "
                        "unchanged scene."
                    )
                    ended_no_grasp_after_fallback = True
                    break

                used_full_scene_fallback = True

                print(
                    "Using full-scene fallback grasps for final selection."
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

            target_region_ply_path = (
                save_grasp_set_debug_ply(
                    env=env,
                    grasps=candidate_grasps,
                    scene_points=full_scene_debug_points,
                    scene_colors=full_scene_debug_colors,
                    output_dir=GRASP_DEBUG_TARGET_REGION_DIR,
                    output_name=(
                        f"{target_object}_attempt_{attempt}_"
                        "target_region.ply"
                    ),
                )
            )

            # Diagnostic only: show every grasp retained inside the DINO target
            # region before the final weighted angle / preferred-XY ranking.
            all_candidate_xy_distances = np.linalg.norm(
                candidate_grasps[:, :2]
                - preferred_world_point[None, :2],
                axis=1,
            )

            all_candidate_angle_scores = np.exp(
                -(
                    candidate_angles
                    / float(GRASP_SELECTION_ANGLE_SIGMA_DEG)
                ) ** 2
            )

            all_candidate_preferred_scores = np.exp(
                -(
                    all_candidate_xy_distances
                    / float(GRASP_SELECTION_PREFERRED_SIGMA_M)
                ) ** 2
            )

            if used_full_scene_fallback:
                # Fallback candidates are ranked for scene perturbation, not for
                # target preferred-location fidelity.
                all_candidate_final_scores = (
                    float(FULL_SCENE_FALLBACK_ANGLE_WEIGHT)
                    * all_candidate_angle_scores
                    + float(FULL_SCENE_FALLBACK_GRASPNET_WEIGHT)
                    * candidate_scores
                )
            else:
                all_candidate_final_scores = (
                    float(GRASP_SELECTION_ANGLE_WEIGHT)
                    * all_candidate_angle_scores
                    + float(GRASP_SELECTION_PREFERRED_WEIGHT)
                    * all_candidate_preferred_scores
                )

            print(
                "Grasp source:",
                "full-scene fallback" if used_full_scene_fallback else "target crop",
            )

            selection = select_weighted_grasp(
                grasps=candidate_grasps,
                scores=candidate_scores,
                angles_deg=candidate_angles,
                preferred_world_point=(
                    preferred_world_point
                ),
                fallback_mode=used_full_scene_fallback,
            )

            selected_grasp = selection[
                "selected_grasp"
            ]

            selected_grasp_index = int(selection["selected_index"])
            print(
                "Selected grasp:",
                f"index={selected_grasp_index}, "
                f"graspnet={selection['selected_score']:.4f}, "
                f"angle={selection['selected_angle_deg']:.2f} deg, "
                f"xy_dist={selection['selected_xy_distance']:.4f} m, "
                f"final={selection['selected_final_score']:.4f}",
            )

            selected_grasp_angle_deg = float(
                selection["selected_angle_deg"]
            )

            # Diagnostic only: reproduce the pose conversion used by
            # execute_grasp_pose_ik() so we can inspect the selected target
            # before any motion is commanded. The pregrasp offset is WORLD
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

            print(
                "Selected grasp pose:",
                f"center={np.round(selected_grasp[:3], 4).tolist()}, "
                f"approach={np.round(selected_approach_axis, 4).tolist()}, "
                f"pregrasp={np.round(predicted_pregrasp_pose[:3, 3], 4).tolist()}",
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

            # ------------------------------------------------------------
            # 5. Grasp Execution and Recovery
            # ------------------------------------------------------------

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
            )

            # ---------------------------------------------------------
            # IK FAILURE RECOVERY (JOINT_POSITION ONLY)
            #
            # If the selected grasp is not IK-converged / executable,
            # reject this perception cycle, return to the saved home joint
            # configuration using q_ref + JOINT_POSITION, then re-perceive.
            # ---------------------------------------------------------
            if not bool(execution.get("success", False)):
                failed_phase = execution.get("failed_phase")

                reward = -1.0
                episode_reward += reward

                print(
                    f"Reward: {reward:+.4f} "
                    f"(grasp execution failed), "
                    f"episode_reward={episode_reward:+.4f}"
                )

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

                # Recovery policy:
                #   - if lift IK fails after the gripper has closed, release
                #     the object immediately at the current pose, then return
                #     home with an open gripper;
                #   - failures before lift return home with the gripper open.
                if failed_phase == "lift":
                    print(
                        "Lift failed after gripper close. "
                        "Releasing object at current pose before return-home."
                    )
                    env.open_gripper(steps=30)
                    recorder.capture_frame()
                    recorder.add_hold(0.5)

                recovery_result = env.move_joints_qref(
                    target_joint_positions=home_joint_positions,
                    gripper_command=-1.0,
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
                    "Return-home recovery failed. Stopping this run."
                )
                recorder.capture_frame()
                recorder.add_hold(2.0)
                break

            # =========================================================
            # Grasp / transport task semantics.
            #
            # Physical grasp success and task-target success are separate:
            #   - gripper width answers "are we still holding something?"
            #   - after release, simulator-side evaluation checks whether the
            #     task target is inside the bin.
            #
            # Any held object is transported to the bin. Only AFTER release
            # and JOINT_POSITION return-home do we evaluate task success.
            # =========================================================
            first_gripper_width = env.get_gripper_width()

            gripper_holds_something = bool(
                execution["success"]
                and first_gripper_width
                >= MIN_GRASPED_GRIPPER_WIDTH
            )

            print(
                "Grasp execution:",
                f"motion_success={execution['success']}, "
                f"gripper_width={first_gripper_width:.4f}, "
                f"hold={gripper_holds_something}",
            )

            for phase_name in ["pregrasp", "grasp", "lift"]:
                phase_result = execution.get(phase_name)
                if phase_result is None:
                    continue

                phase_diagnostic = summarize_ik_phase_diagnostic(phase_result)
                position_error = phase_result.get("position_error_norm")
                position_error_text = (
                    "n/a"
                    if position_error is None
                    else f"{1000.0 * float(position_error):.2f} mm"
                )

                print(
                    f"{phase_name}: "
                    f"success={phase_result.get('success')}, "
                    f"pos_err={position_error_text}, "
                    f"steps={phase_diagnostic['physics_steps']}, "
                    f"force_stop={phase_diagnostic['force_stop_triggered']}"
                )

                if (
                    phase_diagnostic["force_stop_triggered"]
                    or not bool(phase_result.get("success", False))
                ):
                    print(
                        f"  {phase_name} diagnostics: "
                        f"max_force={phase_diagnostic['max_eef_force_metric_seen']}, "
                        f"max_tracking={phase_diagnostic['max_tracking_error_seen']}, "
                        f"max_torque={phase_diagnostic['max_raw_torque_seen']}"
                    )
                    if phase_diagnostic["diagnostic_contact_pairs"]:
                        print(
                            f"  {phase_name} contacts:",
                            phase_diagnostic["diagnostic_contact_pairs"],
                        )
                    if phase_diagnostic["force_stop_triggered"]:
                        print(
                            f"  {phase_name} force-stop metric:",
                            phase_diagnostic["force_stop_metric"],
                        )

            # Physical grasp failed: no object is retained by the gripper.
            # Recovery uses JOINT_POSITION only.
            if not gripper_holds_something:
                print()
                print(
                    "Physical grasp failed: gripper-width check indicates "
                    "that no object is being held."
                )

                reward = -1.0
                episode_reward += reward

                print(
                    f"Reward: {reward:+.4f} "
                    f"(empty grasp), "
                    f"episode_reward={episode_reward:+.4f}"
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
                        "Return-home recovery failed. Stopping this run."
                    )
                    break

                env.open_gripper(steps=30)
                recorder.capture_frame()
                recorder.add_hold(0.5)

                print(
                    "Home reached. Re-perceiving the current scene."
                )
                continue

            # PyBullet-style grasped-object identification after lift.
            grasped_obj_id = None
            grasped_obj_name = None
            max_height = -np.inf

            for object_name, body_id in env.object_body_ids.items():
                object_position = np.asarray(
                    env.sim.data.body_xpos[int(body_id)],
                    dtype=np.float64,
                )

                if float(object_position[2]) >= max_height:
                    max_height = float(object_position[2])
                    grasped_obj_id = int(body_id)
                    grasped_obj_name = object_name

            grasped_object_position = np.asarray(
                env.sim.data.body_xpos[grasped_obj_id],
                dtype=np.float64,
            )

            target_position_after_lift = np.asarray(
                env.sim.data.body_xpos[target_body_id],
                dtype=np.float64,
            )

            pos_dist = float(
                np.linalg.norm(
                    grasped_object_position
                    - target_position_after_lift
                )
            )

            print(
                "Grasped object:",
                f"{grasped_obj_name}, "
                f"target_distance={pos_dist:.4f} m",
            )

            # ------------------------------------------------------------
            # 6. Transport and Task Evaluation
            # ------------------------------------------------------------

            print("Transporting held object to the bin.")

            recorder.capture_frame()
            recorder.add_hold(0.5)

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
                "Transport:",
                f"success={drop_motion_result['success']}, reason={drop_motion_result['failure_reason']}",
            )

            if not drop_motion_result["success"]:
                print(
                    "Fixed drop-joint transport failed."
                )

                reward = -1.0
                episode_reward += reward

                print(
                    f"Reward: {reward:+.4f} "
                    f"(transport failed), "
                    f"episode_reward={episode_reward:+.4f}"
                )
                break

            # ---------------------------------------------------------
            # Second gripper-state check:
            # after reaching the drop/bin pose, immediately before release.
            # ---------------------------------------------------------
            width_before_release = env.get_gripper_width()
            still_holding_at_bin = bool(
                width_before_release
                >= MIN_GRASPED_GRIPPER_WIDTH
            )

            print(
                "Hold at bin:",
                f"{still_holding_at_bin} (width={width_before_release:.4f})",
            )

            if not still_holding_at_bin:
                print(
                    "Held object was lost during transport. Returning home "
                    "without issuing an intentional bin-release command."
                )

                reward = -1.0
                episode_reward += reward

                print(
                    f"Reward: {reward:+.4f} "
                    f"(object lost during transport), "
                    f"episode_reward={episode_reward:+.4f}"
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

            if grasped_obj_id == target_body_id:
                reward = 2.0
                reward_reason = "correct target"
            else:
                reward = -pos_dist / max_pos_dist
                reward_reason = f"wrong object: {grasped_obj_name}"

            episode_reward += reward

            print(
                f"Reward: {reward:+.4f} "
                f"({reward_reason}), "
                f"episode_reward={episode_reward:+.4f}"
            )

            # Physical transport succeeded. Release the held object in the bin.
            open_result = env.open_gripper(steps=40)
            width_after_release = env.get_gripper_width()

            print(
                "Released object in bin:",
                f"gripper_width={width_after_release:.4f}",
            )

            recorder.capture_frame()
            recorder.add_hold(1.0)

            # Ordering:
            # release -> home -> evaluate target success -> re-perceive if needed.
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

            print("Return home after release:", return_home_result["success"])

            if not return_home_result["success"]:
                print(
                    "Return home after release failed; stopping this run."
                )
                break

            # Final simulator-side task evaluation:
            # do not use the pre-grasp nearest-object bookkeeping to decide
            # which object was actually transported. Instead, directly check
            # whether the task target body is now inside the bin.
            target_position_after_release = np.asarray(
                env.sim.data.body_xpos[target_body_id],
                dtype=np.float64,
            ).copy()

            target_inside_bin = bool(
                abs(
                    float(target_position_after_release[0])
                    - float(env.bin_center[0])
                )
                <= float(env.bin_inner_half_size[0])
                and
                abs(
                    float(target_position_after_release[1])
                    - float(env.bin_center[1])
                )
                <= float(env.bin_inner_half_size[1])
            )

            print(
                "Task evaluation:",
                f"target_inside_bin={target_inside_bin}, "
                f"target_xyz={np.round(target_position_after_release, 4).tolist()}",
            )

            if target_inside_bin:
                target_completed = True
                print(
                    f"Target {target_object} is inside the bin. "
                    "Closed-loop task completed."
                )
                break

            print(
                f"Target {target_object} is not inside the bin. "
                "Robot is home, so re-perceiving the changed scene."
            )
            recorder.capture_frame()
            recorder.add_hold(0.5)
            continue

        print()
        print(
            f"Final episode reward: {episode_reward:+.4f}"
        )

        if not target_completed:
            print()
            print(
                f"Closed-loop task ended after "
                f"{attempts_started} perception attempt(s)."
            )

            if ended_no_grasp_after_fallback:
                print(
                    "Stop reason: target-region planning and "
                    "full-scene fallback produced no usable grasp."
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

                print(
                    "Closed-loop video:",
                    video_path,
                    f"({len(recorder.frames)} frames)",
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

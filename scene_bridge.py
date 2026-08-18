"""MuJoCo scene export utilities for the ThinkGrasp bridge."""

from pathlib import Path

import numpy as np


def export_colored_pointcloud_ply(
    output_path,
    points,
    colors,
):
    """Export an XYZRGB point cloud as an ASCII PLY file."""

    output_path = Path(output_path).expanduser().resolve()

    points = np.asarray(
        points,
        dtype=np.float32,
    ).reshape(-1, 3)

    colors = np.asarray(
        colors,
        dtype=np.float32,
    ).reshape(-1, 3)

    if len(points) != len(colors):
        raise ValueError(
            "PLY points and colors must have the same length: "
            f"{len(points)} != {len(colors)}"
        )

    if not np.isfinite(points).all():
        raise ValueError(
            "PLY point cloud contains non-finite XYZ values."
        )

    colors = np.clip(colors, 0.0, 1.0)
    colors_uint8 = np.rint(
        colors * 255.0
    ).astype(np.uint8)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="ascii",
        newline="\n",
    ) as ply_file:
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write(
            f"element vertex {len(points)}\n"
        )
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        ply_file.write("property uchar red\n")
        ply_file.write("property uchar green\n")
        ply_file.write("property uchar blue\n")
        ply_file.write("end_header\n")

        for point, color in zip(
            points,
            colors_uint8,
        ):
            ply_file.write(
                f"{point[0]:.8f} "
                f"{point[1]:.8f} "
                f"{point[2]:.8f} "
                f"{int(color[0])} "
                f"{int(color[1])} "
                f"{int(color[2])}\n"
            )

    return output_path


# -------------------------------------------------------------------------
# Final fixed workspace calibration
# -------------------------------------------------------------------------
# These values are valid for:
#   table centre X = +0.07 m
#   topview position = (0.07, 0.0, 2.23)
#   topview resolution = 640 x 640
#   Panda at the reset / home pose
#
# The perception crop was calibrated once from the top-view segmentation:
#   table image xyxy       = [10, 9, 631, 630]
#   robot rightmost pixel  = 120
#   final fixed crop xyxy  = [150, 95, 500, 545]
#
# GroundingDINO and the future VLM always receive this fixed crop. It is no
# longer recomputed from segmentation during normal closed-loop execution.
TOPVIEW_WIDTH = 640
TOPVIEW_HEIGHT = 640

FIXED_PERCEPTION_CROP_XYXY = np.array(
    [180, 90, 540, 560],
    dtype=np.int32,
)

# -------------------------------------------------------------------------
# Official MuJoCo workspace definition
# -------------------------------------------------------------------------
# This is the FINAL calibrated workspace after the historical Panda-side
# X-min 5 cm adjustment. That adjustment is now absorbed permanently into
# the workspace definition; it is no longer applied as a runtime patch.
#
# Project-wide formal workspace:
#   X = [-0.05249527,  0.34106400]
#   Y = [-0.28215298,  0.29693829]
#
# Perception, heightmap generation, GraspNet and full-scene fallback all use
# exactly this one fixed workspace.
OFFICIAL_WORKSPACE_XY_BOUNDS = np.array(
    [
        [-0.05249527, 0.34106400],
        [-0.28215298, 0.29693829],
    ],
    dtype=np.float64,
)

# Initial settled clutter must remain inside a centred, geometrically similar
# subset of the official workspace. 0.85 preserves the workspace aspect ratio
# while keeping objects away from perception-image edges.
SCENE_VALID_WORKSPACE_SCALE = 0.85

# Z limits remain a calibrated world-frame vertical range. They are combined
# with the runtime-derived official XY bounds by ThinkGraspMinimalEnv.
OFFICIAL_WORKSPACE_Z_LIMITS = np.array(
    [0.82, 1.08],
    dtype=np.float64,
)

# Legacy default retained only for backward-compatible utility calls that may
# still be used outside the formal closed loop. The formal MuJoCo pipeline
# MUST use env.perception_workspace_limits instead.
DEFAULT_WORKSPACE_LIMITS = np.array(
    [
        [-0.155, 0.295],
        [-0.225, 0.225],
        [0.82, 1.08],
    ],
    dtype=np.float64,
)

# Calibrated tabletop plane in the world coordinate system.
# In the real system this value must come from camera / table calibration,
# not from object ground-truth information.
DEFAULT_TABLE_HEIGHT = 0.855

# Remove points on or extremely close to the tabletop.
DEFAULT_TABLE_CLEARANCE = 0.003

# Object initialization no longer uses a rectangular placement range.
# Objects are dropped sequentially above the world-space centre of the
# fixed perception crop; see ThinkGraspMinimalEnv._build_initial_clutter_by_center_drop().


def crop_camera_data_to_workspace(
    camera_data,
    workspace_limits=DEFAULT_WORKSPACE_LIMITS,
    object_body_ids=None,
    robot_safety_margin_pixels=None,
    table_edge_padding_pixels=None,
    crop_xyxy=FIXED_PERCEPTION_CROP_XYXY,
):
    """Apply the final calibrated fixed top-view perception crop.

    ``object_body_ids``, ``robot_safety_margin_pixels`` and
    ``table_edge_padding_pixels`` are retained only for backward-compatible
    calls. Normal execution does not inspect segmentation to locate the robot.

    The fixed crop is valid only for a 640 x 640 top-view image with the
    calibrated table, camera and Panda home pose.
    """

    del object_body_ids
    del robot_safety_margin_pixels
    del table_edge_padding_pixels

    workspace_limits = np.asarray(
        workspace_limits,
        dtype=np.float64,
    )

    if workspace_limits.shape != (3, 2):
        raise ValueError(
            "workspace_limits must have shape (3, 2), "
            f"got {workspace_limits.shape}"
        )

    color = np.asarray(camera_data["color"])
    depth = np.asarray(
        camera_data["depth"],
        dtype=np.float32,
    )
    segmentation = np.asarray(
        camera_data["segmentation"],
        dtype=np.int32,
    )
    pointcloud = np.asarray(
        camera_data["pointcloud"],
        dtype=np.float32,
    )

    if color.shape[:2] != depth.shape:
        raise RuntimeError(
            "RGB and depth shapes do not match: "
            f"{color.shape}, {depth.shape}"
        )

    if segmentation.shape != depth.shape:
        raise RuntimeError(
            "Segmentation and depth shapes do not match: "
            f"{segmentation.shape}, {depth.shape}"
        )

    if pointcloud.shape != (*depth.shape, 3):
        raise RuntimeError(
            f"Unexpected point-cloud shape: {pointcloud.shape}"
        )

    image_height, image_width = depth.shape

    if (
        image_width != TOPVIEW_WIDTH
        or image_height != TOPVIEW_HEIGHT
    ):
        raise RuntimeError(
            "The fixed perception crop was calibrated for "
            f"{TOPVIEW_WIDTH}x{TOPVIEW_HEIGHT}, but received "
            f"{image_width}x{image_height}."
        )

    crop_xyxy = np.asarray(
        crop_xyxy,
        dtype=np.int32,
    ).reshape(4)

    x1, y1, x2, y2 = [
        int(value)
        for value in crop_xyxy
    ]

    if not (
        0 <= x1 < x2 <= image_width
        and 0 <= y1 < y2 <= image_height
    ):
        raise ValueError(
            "Fixed crop is outside the top-view image: "
            f"{crop_xyxy}"
        )

    cropped_intrinsics = np.asarray(
        camera_data["intrinsics"],
        dtype=np.float64,
    ).copy()
    cropped_intrinsics[0, 2] -= x1
    cropped_intrinsics[1, 2] -= y1

    return {
        "color": color[y1:y2, x1:x2].copy(),
        "depth": depth[y1:y2, x1:x2].copy(),
        "segmentation": segmentation[y1:y2, x1:x2].copy(),
        "pointcloud": pointcloud[y1:y2, x1:x2].copy(),
        "intrinsics": cropped_intrinsics,
        "extrinsics": np.asarray(
            camera_data["extrinsics"],
            dtype=np.float64,
        ).copy(),
        "camera_name": str(
            camera_data.get("camera_name", "topview")
        ),
        "full_image_crop_xyxy": crop_xyxy.copy(),
        "workspace_limits": workspace_limits.copy(),
    }



def build_pybullet_style_heightmap(
    topview_data,
    workspace_limits=DEFAULT_WORKSPACE_LIMITS,
    pixel_size=0.002,
):
    """Reconstruct the original ThinkGrasp-style orthographic RGB heightmap.

    Source logic:
        perspective RGB-D/world points
        -> top-down workspace heightmap
        -> VLM / GroundingDINO operate on this orthographic representation.

    No segmentation or simulator object IDs are used.
    """
    workspace_limits = np.asarray(workspace_limits, dtype=np.float64)
    if workspace_limits.shape != (3, 2):
        raise ValueError(
            "workspace_limits must have shape (3, 2), "
            f"got {workspace_limits.shape}"
        )

    pixel_size = float(pixel_size)
    if pixel_size <= 0.0:
        raise ValueError("pixel_size must be positive.")

    points_image = np.asarray(topview_data["pointcloud"], dtype=np.float32)
    color_image = np.asarray(topview_data["color"])

    if points_image.ndim != 3 or points_image.shape[2] != 3:
        raise ValueError(
            "topview pointcloud must have shape (H, W, 3), "
            f"got {points_image.shape}"
        )
    if color_image.shape[:2] != points_image.shape[:2]:
        raise ValueError("topview color / pointcloud shapes do not match.")

    points = points_image.reshape(-1, 3)
    colors = color_image.reshape(-1, color_image.shape[-1])

    valid = (
        np.isfinite(points).all(axis=1)
        & (points[:, 0] >= workspace_limits[0, 0])
        & (points[:, 0] < workspace_limits[0, 1])
        & (points[:, 1] >= workspace_limits[1, 0])
        & (points[:, 1] < workspace_limits[1, 1])
        & (points[:, 2] >= workspace_limits[2, 0])
        & (points[:, 2] < workspace_limits[2, 1])
    )

    points = points[valid]
    colors = colors[valid]

    map_width = int(np.round(
        (workspace_limits[0, 1] - workspace_limits[0, 0]) / pixel_size
    ))
    map_height = int(np.round(
        (workspace_limits[1, 1] - workspace_limits[1, 0]) / pixel_size
    ))

    heightmap = np.zeros((map_height, map_width), dtype=np.float32)
    colormap = np.zeros((map_height, map_width, 3), dtype=np.uint8)
    world_pointmap = np.zeros((map_height, map_width, 3), dtype=np.float32)

    if len(points) > 0:
        # Same z-buffer idea as original utils.get_heightmap(): low -> high,
        # so the highest point overwrites lower points in each orthographic cell.
        order = np.argsort(points[:, 2])
        points = points[order]
        colors = colors[order]

        px = np.floor(
            (points[:, 0] - workspace_limits[0, 0]) / pixel_size
        ).astype(np.int32)
        py = np.floor(
            (points[:, 1] - workspace_limits[1, 0]) / pixel_size
        ).astype(np.int32)

        px = np.clip(px, 0, map_width - 1)
        py = np.clip(py, 0, map_height - 1)

        # Rectangular-workspace adaptation:
        # NumPy image arrays use [row, col] = [y, x].
        # The original ThinkGrasp code used [px, py], which was harmless for
        # its square workspace but becomes invalid once MuJoCo uses a
        # rectangular workspace. Keep the same orthographic mapping while
        # using conventional image indexing so X maps to image columns and
        # Y maps to image rows.
        heightmap[py, px] = points[:, 2] - workspace_limits[2, 0]

        if np.issubdtype(colors.dtype, np.integer):
            rgb = np.clip(colors[:, :3], 0, 255).astype(np.uint8)
        else:
            rgb = colors[:, :3].astype(np.float32)
            if rgb.max(initial=0.0) <= 1.0:
                rgb *= 255.0
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        colormap[py, px] = rgb
        world_pointmap[py, px] = points

    return {
        "color": colormap,
        "depth": heightmap,
        "pointcloud": world_pointmap,
        "workspace_limits": workspace_limits.copy(),
        "pixel_size": pixel_size,
        "camera_name": "pybullet_style_heightmap",
        "source_camera_name": str(topview_data.get("camera_name", "topview")),
        "valid_workspace_point_count": int(len(points)),
        "image_width": int(map_width),
        "image_height": int(map_height),
    }


def export_pybullet_style_target_scene(
    output_path,
    heightmap_data,
    bbox_xyxy,
    workspace_limits=DEFAULT_WORKSPACE_LIMITS,
    crop_margin=20,
):
    """Mirror the original simulation_main.py::crop_pointcloud() logic.

    GroundingDINO bbox (+ source 20-pixel margin)
        -> positive orthographic heights inside bbox
        -> linear image-to-workspace XY conversion
        -> full-workspace XY grid
        -> outside bbox stays at workspace Z-min
        -> bbox cells receive measured Z and RGB

    The original PyBullet workspace Z-min is approximately zero. MuJoCo is
    vertically shifted, so Z-min is added back to keep output in MuJoCo world
    coordinates. This is a coordinate-frame adaptation only.
    """
    output_path = Path(output_path).expanduser().resolve()
    workspace_limits = np.asarray(workspace_limits, dtype=np.float64)

    color_image = np.asarray(heightmap_data["color"])
    depth_image = np.asarray(heightmap_data["depth"], dtype=np.float32)
    if color_image.shape[:2] != depth_image.shape:
        raise ValueError("heightmap RGB / depth shapes do not match.")

    image_height, image_width = depth_image.shape
    raw_bbox = np.asarray(bbox_xyxy, dtype=np.float64).reshape(4)

    x1, y1, x2, y2 = raw_bbox
    crop_margin = int(crop_margin)

    x1 = int(np.floor(x1 - crop_margin))
    y1 = int(np.floor(y1 - crop_margin))
    x2 = int(np.ceil(x2 + crop_margin))
    y2 = int(np.ceil(y2 + crop_margin))

    x1 = int(np.clip(x1, 0, image_width - 1))
    y1 = int(np.clip(y1, 0, image_height - 1))
    x2 = int(np.clip(x2, x1 + 1, image_width))
    y2 = int(np.clip(y2, y1 + 1, image_height))
    expanded_bbox = np.asarray([x1, y1, x2, y2], dtype=np.int32)

    depth_crop = depth_image[y1:y2, x1:x2]
    color_crop = color_image[y1:y2, x1:x2]
    mask = depth_crop > 0.0

    local_y, local_x = np.nonzero(mask)
    if len(local_x) == 0:
        raise RuntimeError(
            "PyBullet-style target crop contains no positive heightmap "
            "pixels inside the expanded GroundingDINO bbox."
        )

    image_x = local_x + x1
    image_y = local_y + y1
    height_values = depth_crop[mask].astype(np.float64)
    measured_colors = color_crop[mask, :3]

    if np.issubdtype(measured_colors.dtype, np.integer):
        measured_colors = measured_colors.astype(np.float32) / 255.0
    else:
        measured_colors = measured_colors.astype(np.float32)
        if measured_colors.max(initial=0.0) > 1.0:
            measured_colors /= 255.0

    # Same image_to_workspace() mapping as the PyBullet source.
    workspace_x = (
        workspace_limits[0, 0]
        + (image_x / float(image_width))
        * (workspace_limits[0, 1] - workspace_limits[0, 0])
    )
    workspace_y = (
        workspace_limits[1, 0]
        + (image_y / float(image_height))
        * (workspace_limits[1, 1] - workspace_limits[1, 0])
    )
    workspace_z = workspace_limits[2, 0] + height_values

    grid_x, grid_y = np.meshgrid(
        np.linspace(workspace_limits[0, 0], workspace_limits[0, 1], image_width),
        np.linspace(workspace_limits[1, 0], workspace_limits[1, 1], image_height),
    )
    grid_z = np.full_like(
        grid_x,
        float(workspace_limits[2, 0]),
        dtype=np.float64,
    )

    full_workspace_points = np.stack(
        (grid_x, grid_y, grid_z),
        axis=-1,
    ).reshape(-1, 3)
    full_colors = np.zeros(
        (full_workspace_points.shape[0], 3),
        dtype=np.float32,
    )

    # Rectangular-workspace adaptation:
    # full_workspace_points comes from a conventional (H, W) meshgrid, so
    # flattening uses row-major index = y * width + x. The original source's
    # x * height + y form was equivalent only for its square-map assumptions.
    grid_indices = (
        image_y.astype(np.int64) * int(image_width)
        + image_x.astype(np.int64)
    )
    valid_indices = (
        (grid_indices >= 0)
        & (grid_indices < len(full_workspace_points))
    )

    full_workspace_points[grid_indices[valid_indices], 2] = (
        workspace_z[valid_indices]
    )
    full_colors[grid_indices[valid_indices]] = measured_colors[valid_indices]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        points=full_workspace_points.astype(np.float32),
        colors=full_colors.astype(np.float32),
        bbox_xyxy=raw_bbox.astype(np.float64),
        expanded_bbox_xyxy=expanded_bbox,
        positive_bbox_pixel_count=np.asarray(len(local_x), dtype=np.int64),
        image_width=np.asarray(image_width, dtype=np.int64),
        image_height=np.asarray(image_height, dtype=np.int64),
        workspace_limits=workspace_limits.astype(np.float64),
        workspace_z_min=np.asarray(workspace_limits[2, 0], dtype=np.float64),
        crop_margin=np.asarray(crop_margin, dtype=np.int64),
    )

    measured_xyz = np.column_stack(
        [workspace_x, workspace_y, workspace_z]
    )

    return {
        "output_path": output_path,
        "point_count": int(len(full_workspace_points)),
        "positive_bbox_pixel_count": int(len(local_x)),
        "bbox_xyxy": raw_bbox.copy(),
        "expanded_bbox_xyxy": expanded_bbox.copy(),
        "measured_xyz_min": measured_xyz.min(axis=0),
        "measured_xyz_max": measured_xyz.max(axis=0),
        "workspace_z_min": float(workspace_limits[2, 0]),
        "image_width": int(image_width),
        "image_height": int(image_height),
        "crop_margin": int(crop_margin),
    }

def export_current_scene(
    env,
    output_path,
    camera_name="topview",
    width=640,
    height=640,
    workspace_limits=DEFAULT_WORKSPACE_LIMITS,
    crop_box_xyxy=None,
    crop_margin=20,
    camera_data=None,
    table_height=DEFAULT_TABLE_HEIGHT,
    table_clearance=DEFAULT_TABLE_CLEARANCE,
):
    """Export an RGB-D point cloud for the GraspNet worker.

    ``camera_data`` may already be the fixed workspace crop returned by
    :func:`crop_camera_data_to_workspace`. A GroundingDINO bbox is then
    applied directly in that cropped top-view image, keeping RGB, depth,
    segmentation and world-frame points perfectly aligned.
    """

    output_path = Path(
        output_path
    ).expanduser().resolve()

    workspace_limits = np.asarray(
        workspace_limits,
        dtype=np.float64,
    )

    if workspace_limits.shape != (3, 2):
        raise ValueError(
            "workspace_limits must have shape (3, 2), "
            f"got {workspace_limits.shape}"
        )

    table_height = float(table_height)
    table_clearance = float(table_clearance)

    if table_clearance < 0.0:
        raise ValueError(
            "table_clearance must be non-negative, "
            f"got {table_clearance}"
        )

    if camera_data is None:
        camera_data = env.get_camera_data(
            camera_name=camera_name,
            width=width,
            height=height,
        )

    color = np.asarray(camera_data["color"])
    depth = np.asarray(
        camera_data["depth"],
        dtype=np.float32,
    )
    segmentation = np.asarray(
        camera_data["segmentation"],
        dtype=np.int32,
    )
    pointcloud = np.asarray(
        camera_data["pointcloud"],
        dtype=np.float32,
    )

    if color.shape[:2] != depth.shape:
        raise RuntimeError(
            "RGB and depth shapes do not match: "
            f"{color.shape}, {depth.shape}"
        )

    if segmentation.shape != depth.shape:
        raise RuntimeError(
            "Segmentation and depth shapes do not match: "
            f"{segmentation.shape}, {depth.shape}"
        )

    if pointcloud.shape != (*depth.shape, 3):
        raise RuntimeError(
            f"Unexpected point-cloud shape: {pointcloud.shape}"
        )

    image_height, image_width = depth.shape

    if crop_box_xyxy is None:
        bbox_mask = np.ones(depth.shape, dtype=bool)
        expanded_crop_box = None
    else:
        crop_box_xyxy = np.asarray(
            crop_box_xyxy,
            dtype=np.float64,
        ).reshape(4)

        x1, y1, x2, y2 = crop_box_xyxy

        x1 = int(np.floor(x1 - crop_margin))
        y1 = int(np.floor(y1 - crop_margin))
        x2 = int(np.ceil(x2 + crop_margin))
        y2 = int(np.ceil(y2 + crop_margin))

        x1 = int(np.clip(x1, 0, image_width - 1))
        y1 = int(np.clip(y1, 0, image_height - 1))
        x2 = int(np.clip(x2, x1 + 1, image_width))
        y2 = int(np.clip(y2, y1 + 1, image_height))

        bbox_mask = np.zeros(depth.shape, dtype=bool)
        bbox_mask[y1:y2, x1:x2] = True

        expanded_crop_box = np.array(
            [x1, y1, x2, y2],
            dtype=np.int32,
        )

    # Do not use MuJoCo body IDs to decide which points are graspable.
    # The real system cannot directly obtain simulator object identities.
    # Fixed workspace filtering disabled on purpose.
    # We keep the function signature unchanged for compatibility,
    # but no longer remove points by a global XYZ box here.
    workspace_mask = np.ones(
        pointcloud.shape[:2],
        dtype=bool,
    )

    # This is a calibrated geometric condition that can also be used
    # in a real RGB-D setup. It removes the tabletop without requiring
    # simulator segmentation or object IDs.
    above_table_mask = (
        pointcloud[..., 2]
        >= table_height + table_clearance
    )

    # Points that are geometrically valid inside the GroundingDINO bbox,
    # scene objects inside the GroundingDINO candidate bbox, before the
    # 3D workspace restriction is applied.
    pre_workspace_mask = (
        np.isfinite(pointcloud).all(axis=-1)
        & np.isfinite(depth)
        & (depth > 0.0)
        & (depth < 3.0)
        & bbox_mask
    )

    # Final GraspNet input: keep only points that also lie inside the
    # calibrated world-coordinate workspace.
    valid_mask = (
        pre_workspace_mask
        & workspace_mask
        & above_table_mask
    )

    pre_workspace_point_count = int(
        np.count_nonzero(pre_workspace_mask)
    )
    workspace_point_count = int(
        np.count_nonzero(valid_mask)
    )

    if pre_workspace_point_count > 0:
        workspace_point_ratio = (
            workspace_point_count
            / pre_workspace_point_count
        )
    else:
        workspace_point_ratio = 0.0

    points = pointcloud[valid_mask]
    colors = color[valid_mask]
    body_ids = segmentation[valid_mask]

    if len(points) == 0:
        raise RuntimeError(
            "No valid points remained after applying the "
            "GroundingDINO bbox and 3D workspace filter."
        )

    if np.issubdtype(colors.dtype, np.integer):
        colors_float = (
            colors.astype(np.float32) / 255.0
        )
    else:
        colors_float = colors.astype(np.float32)

        if colors_float.max(initial=0.0) > 1.0:
            colors_float /= 255.0

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        output_path,
        points=points.astype(np.float32),
        colors=colors_float.astype(np.float32),
        depth=depth,
        color=color,
        intrinsics=np.asarray(
            camera_data["intrinsics"],
            dtype=np.float64,
        ),
        extrinsics=np.asarray(
            camera_data["extrinsics"],
            dtype=np.float64,
        ),
        camera_name=np.array(camera_name),
        workspace_limits=workspace_limits,
        table_height=np.asarray(table_height, dtype=np.float64),
        table_clearance=np.asarray(
            table_clearance,
            dtype=np.float64,
        ),
        crop_box_xyxy=(
            np.asarray(crop_box_xyxy, dtype=np.float64)
            if crop_box_xyxy is not None
            else np.empty((0,), dtype=np.float64)
        ),
        expanded_crop_box_xyxy=(
            expanded_crop_box
            if expanded_crop_box is not None
            else np.empty((0,), dtype=np.int32)
        ),
        crop_margin=np.asarray(crop_margin),
        full_image_crop_xyxy=np.asarray(
            camera_data.get(
                "full_image_crop_xyxy",
                np.empty((0,), dtype=np.int32),
            ),
            dtype=np.int32,
        ),
    )

    return {
        "output_path": output_path,
        "point_count": int(len(points)),
        "pre_workspace_point_count": pre_workspace_point_count,
        "workspace_point_count": workspace_point_count,
        "workspace_point_ratio": float(workspace_point_ratio),
        "table_height": table_height,
        "table_clearance": table_clearance,
        "xyz_min": points.min(axis=0),
        "xyz_max": points.max(axis=0),
        "crop_box_xyxy": (
            None
            if crop_box_xyxy is None
            else np.asarray(crop_box_xyxy, dtype=np.float64)
        ),
        "expanded_crop_box_xyxy": expanded_crop_box,
    }


def export_fused_target_scene(
    env,
    output_path,
    topview_data,
    bbox_xyxy,
    ply_output_path=None,
    camera_configs=(
        ("topview", 640, 640),
        ("frontview", 640, 480),
        ("left_oblique_25deg", 640, 480),
        ("right_oblique_25deg", 640, 480),
    ),
    workspace_limits=DEFAULT_WORKSPACE_LIMITS,
    table_height=DEFAULT_TABLE_HEIGHT,
    table_clearance=DEFAULT_TABLE_CLEARANCE,
    bbox_xy_padding=0.015,
    voxel_size=0.003,
):
    """Export a target crop from a four-camera fused point cloud.

    Pipeline:
        topview RGB bbox
        -> measured topview points inside bbox
        -> world-frame XY target bounds
        -> symmetric four-camera world-frame point-cloud fusion
        -> target XY crop
        -> voxel deduplication

    No simulator object IDs, segmentation labels, object centres, or
    ground-truth object poses are used.
    """

    output_path = Path(
        output_path
    ).expanduser().resolve()

    workspace_limits = np.asarray(
        workspace_limits,
        dtype=np.float64,
    )

    if workspace_limits.shape != (3, 2):
        raise ValueError(
            "workspace_limits must have shape (3, 2), "
            f"got {workspace_limits.shape}"
        )

    bbox_xyxy = np.asarray(
        bbox_xyxy,
        dtype=np.float64,
    ).reshape(4)

    table_height = float(table_height)
    table_clearance = float(table_clearance)
    bbox_xy_padding = float(bbox_xy_padding)
    voxel_size = float(voxel_size)

    if table_clearance < 0.0:
        raise ValueError("table_clearance must be non-negative.")

    if bbox_xy_padding < 0.0:
        raise ValueError("bbox_xy_padding must be non-negative.")

    if voxel_size <= 0.0:
        raise ValueError("voxel_size must be positive.")

    def valid_geometric_mask(points, depth):
        # Fixed global XYZ workspace filtering is temporarily disabled.
        # Keep only valid depth points above the calibrated tabletop.
        return (
            np.isfinite(points).all(axis=-1)
            & np.isfinite(depth)
            & (depth > 0.0)
            & (depth < 3.0)
            & (
                points[..., 2]
                >= table_height + table_clearance
            )
        )

    # --------------------------------------------------------------
    # 1. Convert the top-view GroundingDINO bbox to world XY bounds.
    # --------------------------------------------------------------
    topview_points = np.asarray(
        topview_data["pointcloud"],
        dtype=np.float32,
    )
    topview_depth = np.asarray(
        topview_data["depth"],
        dtype=np.float32,
    )

    image_height, image_width = topview_depth.shape

    x1, y1, x2, y2 = bbox_xyxy

    x1 = int(np.clip(np.floor(x1), 0, image_width - 1))
    y1 = int(np.clip(np.floor(y1), 0, image_height - 1))
    x2 = int(np.clip(np.ceil(x2), x1 + 1, image_width))
    y2 = int(np.clip(np.ceil(y2), y1 + 1, image_height))

    bbox_points = topview_points[y1:y2, x1:x2]
    bbox_depth = topview_depth[y1:y2, x1:x2]

    bbox_valid_mask = valid_geometric_mask(
        bbox_points,
        bbox_depth,
    )

    measured_bbox_points = bbox_points[
        bbox_valid_mask
    ]

    if len(measured_bbox_points) == 0:
        raise RuntimeError(
            "No valid measured top-view points remained "
            "inside the GroundingDINO bbox."
        )

    target_xy_min = (
        measured_bbox_points[:, :2].min(axis=0)
        - bbox_xy_padding
    )
    target_xy_max = (
        measured_bbox_points[:, :2].max(axis=0)
        + bbox_xy_padding
    )

    # Do not clip the measured target XY bounds to the fixed workspace.
    # The bounds now come only from the measured GroundingDINO bbox
    # points plus bbox_xy_padding.

    # --------------------------------------------------------------
    # 2. Collect geometrically valid world-frame points from 4 cameras.
    # --------------------------------------------------------------
    fused_points_parts = []
    fused_colors_parts = []
    camera_point_counts = []

    for camera_name, width, height in camera_configs:
        if camera_name == "topview":
            camera_data = topview_data
        else:
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
        color = np.asarray(
            camera_data["color"],
        )

        valid_mask = valid_geometric_mask(
            points_image,
            depth,
        )

        points = points_image[valid_mask]
        colors = color[valid_mask]

        if np.issubdtype(colors.dtype, np.integer):
            colors = (
                colors.astype(np.float32) / 255.0
            )
        else:
            colors = colors.astype(np.float32)

            if colors.max(initial=0.0) > 1.0:
                colors /= 255.0

        fused_points_parts.append(points)
        fused_colors_parts.append(colors)
        camera_point_counts.append(int(len(points)))

    fused_points_raw = np.concatenate(
        fused_points_parts,
        axis=0,
    )
    fused_colors_raw = np.concatenate(
        fused_colors_parts,
        axis=0,
    )

    # --------------------------------------------------------------
    # 3. Crop the fused cloud with the target world XY bounds.
    # --------------------------------------------------------------
    target_mask = (
        (fused_points_raw[:, 0] >= target_xy_min[0])
        & (fused_points_raw[:, 0] <= target_xy_max[0])
        & (fused_points_raw[:, 1] >= target_xy_min[1])
        & (fused_points_raw[:, 1] <= target_xy_max[1])
    )

    target_points_raw = fused_points_raw[target_mask]
    target_colors_raw = fused_colors_raw[target_mask]

    if len(target_points_raw) == 0:
        raise RuntimeError(
            "No four-camera fused points remained inside "
            "the target world XY region."
        )

    # --------------------------------------------------------------
    # 4. Keep one representative point per voxel.
    # --------------------------------------------------------------
    voxel_indices = np.floor(
        target_points_raw / voxel_size
    ).astype(np.int64)

    _, unique_indices = np.unique(
        voxel_indices,
        axis=0,
        return_index=True,
    )

    unique_indices = np.sort(unique_indices)

    target_points = target_points_raw[unique_indices]
    target_colors = target_colors_raw[unique_indices]

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        output_path,
        points=target_points.astype(np.float32),
        colors=target_colors.astype(np.float32),
        raw_points=target_points_raw.astype(np.float32),
        raw_colors=target_colors_raw.astype(np.float32),
        bbox_xyxy=bbox_xyxy,
        target_xy_min=target_xy_min.astype(np.float64),
        target_xy_max=target_xy_max.astype(np.float64),
        camera_names=np.asarray(
            [config[0] for config in camera_configs]
        ),
        camera_point_counts=np.asarray(
            camera_point_counts,
            dtype=np.int64,
        ),
        workspace_limits=workspace_limits,
        table_height=np.asarray(
            table_height,
            dtype=np.float64,
        ),
        table_clearance=np.asarray(
            table_clearance,
            dtype=np.float64,
        ),
        bbox_xy_padding=np.asarray(
            bbox_xy_padding,
            dtype=np.float64,
        ),
        voxel_size=np.asarray(
            voxel_size,
            dtype=np.float64,
        ),
    )

    if ply_output_path is not None:
        ply_output_path = export_colored_pointcloud_ply(
            output_path=ply_output_path,
            points=target_points,
            colors=target_colors,
        )

    return {
        "output_path": output_path,
        "ply_output_path": ply_output_path,
        "bbox_xyxy": bbox_xyxy.copy(),
        "target_xy_min": target_xy_min,
        "target_xy_max": target_xy_max,
        "topview_bbox_point_count": int(
            len(measured_bbox_points)
        ),
        "camera_point_counts": dict(
            zip(
                [config[0] for config in camera_configs],
                camera_point_counts,
            )
        ),
        "raw_fused_point_count": int(
            len(fused_points_raw)
        ),
        "raw_target_point_count": int(
            len(target_points_raw)
        ),
        "point_count": int(len(target_points)),
        "xyz_min": target_points.min(axis=0),
        "xyz_max": target_points.max(axis=0),
    }

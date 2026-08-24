"""Visualization helpers for closed-loop perception debugging."""

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _to_uint8_rgb(image):
    image = np.asarray(image)

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"Expected RGB image with shape (H, W, 3), got {image.shape}."
        )

    return image


def _load_default_font():
    return ImageFont.load_default()


def _draw_text_with_background(
    draw,
    xy,
    text,
    text_fill,
    background_fill,
    padding=4,
):
    font = _load_default_font()
    left, top, right, bottom = draw.textbbox(
        xy,
        text,
        font=font,
    )
    draw.rectangle(
        [
            left - padding,
            top - padding,
            right + padding,
            bottom + padding,
        ],
        fill=background_fill,
    )
    draw.text(
        xy,
        text,
        fill=text_fill,
        font=font,
    )


def save_vlm_selection_visualization(
    image,
    bbox_xyxy,
    selected_object,
    preferred_location,
    output_path,
    centroid_xy=None,
):
    """Save a visualization of the VLM selection result."""

    image = _to_uint8_rgb(image)
    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    canvas = Image.fromarray(image).convert("RGBA")
    overlay = Image.new(
        "RGBA",
        canvas.size,
        (0, 0, 0, 0),
    )
    draw = ImageDraw.Draw(overlay)

    x1, y1, x2, y2 = [
        int(round(v))
        for v in bbox_xyxy
    ]

    purple = (170, 60, 255, 255)
    purple_fill = (170, 60, 255, 50)
    white = (255, 255, 255, 255)

    draw.rectangle(
        [x1, y1, x2, y2],
        outline=purple,
        width=4,
    )
    draw.rectangle(
        [x1, y1, x2, y2],
        fill=purple_fill,
    )

    # VLM centroid: draw only a red marker on the image.
    # Text is kept in a separate annotation box to avoid clutter.
    if centroid_xy is not None:
        centroid_x, centroid_y = [
            int(round(v))
            for v in centroid_xy
        ]

        centroid_x = max(
            0,
            min(canvas.width - 1, centroid_x),
        )
        centroid_y = max(
            0,
            min(canvas.height - 1, centroid_y),
        )

        centroid_radius = 6

        draw.ellipse(
            [
                centroid_x - centroid_radius,
                centroid_y - centroid_radius,
                centroid_x + centroid_radius,
                centroid_y + centroid_radius,
            ],
            fill=(255, 0, 0, 255),
            outline=white,
            width=2,
        )

        centroid_text = (
            f"({centroid_x}, {centroid_y})"
        )
    else:
        centroid_text = "N/A"

    # Keep semantic annotations away from the centroid marker itself.
    # A fixed information box makes repeated attempts easier to compare.
    info_lines = [
        f"Selected: {selected_object}",
        f"Centroid: {centroid_text}",
        f"Preferred: {preferred_location}",
    ]

    font = _load_default_font()
    info_x = 6
    info_y = 6
    line_gap = 3
    padding = 5

    line_boxes = [
        draw.textbbox(
            (0, 0),
            line,
            font=font,
        )
        for line in info_lines
    ]

    line_widths = [
        box[2] - box[0]
        for box in line_boxes
    ]
    line_heights = [
        box[3] - box[1]
        for box in line_boxes
    ]

    info_width = max(line_widths)
    info_height = (
        sum(line_heights)
        + line_gap * (len(info_lines) - 1)
    )

    draw.rounded_rectangle(
        [
            info_x - padding,
            info_y - padding,
            info_x + info_width + padding,
            info_y + info_height + padding,
        ],
        radius=5,
        fill=(80, 20, 140, 225),
        outline=white,
        width=1,
    )

    current_y = info_y

    for line, line_height in zip(
        info_lines,
        line_heights,
    ):
        draw.text(
            (info_x, current_y),
            line,
            fill=white,
            font=font,
        )

        current_y += (
            line_height
            + line_gap
        )

    merged = Image.alpha_composite(
        canvas,
        overlay,
    ).convert("RGB")
    imageio.imwrite(
        output_path,
        np.asarray(merged),
    )

    return output_path


def save_grounding_grid_visualization(
    image,
    bbox_xyxy,
    preferred_location,
    output_path,
    center_pixel_xy=None,
):
    """Save GroundingDINO bbox + 3x3 grid + highlighted preferred cell."""

    image = _to_uint8_rgb(image)
    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    preferred_location = int(preferred_location)
    if not 1 <= preferred_location <= 9:
        preferred_location = 5

    canvas = Image.fromarray(image).convert("RGBA")
    overlay = Image.new(
        "RGBA",
        canvas.size,
        (0, 0, 0, 0),
    )
    draw = ImageDraw.Draw(overlay)

    x1, y1, x2, y2 = [
        float(v)
        for v in bbox_xyxy
    ]

    outer_color = (40, 220, 255, 255)
    grid_color = (255, 255, 255, 220)
    preferred_outline = (255, 215, 0, 255)
    preferred_fill = (255, 215, 0, 70)
    number_color = (255, 255, 255, 255)
    font = _load_default_font()

    # Outer bbox
    draw.rectangle(
        [x1, y1, x2, y2],
        outline=outer_color,
        width=4,
    )

    cell_width = (x2 - x1) / 3.0
    cell_height = (y2 - y1) / 3.0

    # Grid lines
    for i in range(1, 3):
        xi = x1 + i * cell_width
        yi = y1 + i * cell_height
        draw.line(
            [xi, y1, xi, y2],
            fill=grid_color,
            width=2,
        )
        draw.line(
            [x1, yi, x2, yi],
            fill=grid_color,
            width=2,
        )

    preferred_row = (preferred_location - 1) // 3
    preferred_col = (preferred_location - 1) % 3

    pref_x1 = x1 + preferred_col * cell_width
    pref_x2 = x1 + (preferred_col + 1) * cell_width
    pref_y1 = y1 + preferred_row * cell_height
    pref_y2 = y1 + (preferred_row + 1) * cell_height

    draw.rectangle(
        [pref_x1, pref_y1, pref_x2, pref_y2],
        fill=preferred_fill,
        outline=preferred_outline,
        width=4,
    )

    # Draw numbers 1..9
    number = 1
    for row in range(3):
        for col in range(3):
            cx = x1 + (col + 0.5) * cell_width
            cy = y1 + (row + 0.5) * cell_height
            text = str(number)

            tx0, ty0, tx1, ty1 = draw.textbbox(
                (0, 0),
                text,
                font=font,
            )
            tw = tx1 - tx0
            th = ty1 - ty0

            bg_fill = (
                (120, 90, 0, 220)
                if number == preferred_location
                else (0, 0, 0, 140)
            )

            draw.rounded_rectangle(
                [
                    cx - tw / 2 - 4,
                    cy - th / 2 - 4,
                    cx + tw / 2 + 4,
                    cy + th / 2 + 4,
                ],
                radius=4,
                fill=bg_fill,
            )
            draw.text(
                (cx - tw / 2, cy - th / 2),
                text,
                fill=number_color,
                font=font,
            )
            number += 1

    if center_pixel_xy is not None:
        center_x, center_y = [
            int(v)
            for v in center_pixel_xy
        ]
        radius = 6
        draw.ellipse(
            [
                center_x - radius,
                center_y - radius,
                center_x + radius,
                center_y + radius,
            ],
            fill=(255, 70, 70, 255),
            outline=(255, 255, 255, 255),
            width=2,
        )

    label = (
        f"GroundingDINO 3x3 | preferred={preferred_location}"
    )
    _draw_text_with_background(
        draw,
        (int(round(x1)), max(0, int(round(y1)) - 18)),
        label,
        text_fill=(255, 255, 255, 255),
        background_fill=(0, 80, 120, 235),
    )

    merged = Image.alpha_composite(
        canvas,
        overlay,
    ).convert("RGB")

    imageio.imwrite(
        output_path,
        np.asarray(merged),
    )

    return output_path

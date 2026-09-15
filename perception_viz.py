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
    show_info_box=True,
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
    purple_fill = (170, 60, 255, 0)
    white = (255, 255, 255, 255)

    draw.rectangle(
        [x1, y1, x2, y2],
        outline=purple,
        width=4,
    )
    if purple_fill[3] > 0:
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

    if show_info_box:
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
    score=None,
):
    """Save the selected GroundingDINO bbox visualization.

    The function keeps the original name and unused arguments for backward
    compatibility with run_closed_loop.py, but the output is intentionally
    simplified for thesis figures: selected bbox + optional score only.
    """

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
        float(v)
        for v in bbox_xyxy
    ]

    selected_color = (0, 120, 255, 255)
    score_color = (0, 120, 255, 255)
    font = _load_default_font()

    draw.rectangle(
        [x1, y1, x2, y2],
        outline=selected_color,
        width=4,
    )

    if score is not None:
        score_text = f"{float(score):.3f}"
        text_xy = (
            int(round(x1)),
            max(0, int(round(y1)) - 18),
        )
        draw.text(
            text_xy,
            score_text,
            fill=score_color,
            font=font,
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

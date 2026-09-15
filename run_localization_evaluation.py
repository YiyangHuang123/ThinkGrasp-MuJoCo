import argparse
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from vlm_bridge import run_vlm_selection
from run_closed_loop import run_groundingdino_detection


TARGET_GOALS = {
    "gaming_mouse": "Please pick up the gaming mouse.",
    "mario_figure": "Please pick up the Mario figure.",
    "white_ramekin": "Please pick up the white ramekin.",
}

CONFIDENCE_WEIGHT = 0.70
CENTROID_WEIGHT = 0.30
PIXEL_SIGMA = 80.0


def pixel_point_from_relative(relative_xy, width, height):
    x = int(round(max(0, min(1000, relative_xy[0])) * width / 1000.0))
    y = int(round(max(0, min(1000, relative_xy[1])) * height / 1000.0))
    return x, y


def candidate_score(box, dino_score, centroid_xy):
    x1, y1, x2, y2 = [float(value) for value in box]
    bx = (x1 + x2) / 2.0
    by = (y1 + y2) / 2.0
    distance = math.hypot(bx - centroid_xy[0], by - centroid_xy[1])
    centroid_score = math.exp(-((distance / PIXEL_SIGMA) ** 2))
    final_score = (
        CONFIDENCE_WEIGHT * float(dino_score)
        + CENTROID_WEIGHT * centroid_score
    )
    return distance, centroid_score, final_score


def draw_candidates(
    image,
    boxes,
    display_scores,
    centroid_xy=None,
    selected_index=None,
    selected_only=False,
    show_centroid=True,
):
    canvas = Image.fromarray(np.asarray(image).astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20
        )
    except Exception:
        font = ImageFont.load_default()
    for index, box in enumerate(boxes):
        if selected_only and index != selected_index:
            continue
        x1, y1, x2, y2 = [int(round(value)) for value in box]
        selected = index == selected_index
        color = (0, 120, 255) if selected else (255, 0, 0)
        width = 5 if selected else 2
        draw.rectangle((x1, y1, x2, y2), outline=color, width=width)
        draw.text(
            (max(0, x1), max(0, y1 - 24)),
            f"{float(display_scores[index]):.3f}",
            fill=color,
            font=font,
            stroke_width=2,
            stroke_fill=(255, 255, 255),
        )
    if show_centroid and centroid_xy is not None:
        draw.ellipse(
            (centroid_xy[0] - 5, centroid_xy[1] - 5,
             centroid_xy[0] + 5, centroid_xy[1] + 5),
            fill=(220, 30, 30),
        )
    return np.asarray(canvas)


def run_case(image_path, output_dir):
    image_path = Path(image_path)
    image = imageio.imread(image_path)
    height, width = image.shape[:2]
    target = image_path.name.split("_", 1)[0] + "_" + image_path.name.split("_", 2)[1]
    target = image_path.stem.rsplit("_", 1)[0]
    target = target.rsplit("_", 1)[0] if target.endswith(("clear", "cluttered", "occluded")) else target
    goal = TARGET_GOALS[target]

    vlm = run_vlm_selection(
        image_path=image_path,
        goal=goal,
        system_prompt_path=Path(__file__).resolve().parent / "vlm_system_prompt.txt",
    )
    result = vlm["result"]
    selected_object = result["selected_object"]
    selection_reason = result.get("selection_reason")
    relative_centroid = result["selected_centroid_coordinates_relative"]
    centroid_xy = pixel_point_from_relative(relative_centroid, width, height)
    grounding_prompt = selected_object.replace("_", " ")

    case_dir = output_dir / image_path.stem
    case_dir.mkdir(parents=True, exist_ok=True)
    candidate_viz_path = case_dir / "grounding_candidates.png"
    grounding = run_groundingdino_detection(
        image=image,
        text_prompt=grounding_prompt,
        visualization_path=candidate_viz_path,
    )

    boxes = grounding["boxes"]
    scores = grounding["scores"]
    phrases = grounding["phrases"]
    rankings = []
    for index, (box, score, phrase) in enumerate(zip(boxes, scores, phrases)):
        distance, centroid_score, final_score = candidate_score(
            box, score, centroid_xy
        )
        rankings.append({
            "index": index,
            "box_xyxy": [float(value) for value in box],
            "phrase": str(phrase),
            "dino_score": float(score),
            "pixel_distance": distance,
            "centroid_score": centroid_score,
            "final_score": final_score,
        })
    rankings.sort(key=lambda item: item["final_score"], reverse=True)
    selected_index = rankings[0]["index"] if rankings else None

    final_scores = [item["final_score"] for item in sorted(rankings, key=lambda item: item["index"])]
    dino_scores = [float(score) for score in scores]
    candidate_image = draw_candidates(
        image, boxes, dino_scores, centroid_xy, selected_index=None
    )
    selected_image = draw_candidates(
        image, boxes, final_scores, centroid_xy,
        selected_index=selected_index, selected_only=True
    )
    imageio.imwrite(case_dir / "input_with_centroid.png", image)
    imageio.imwrite(case_dir / "grounding_candidates.png", candidate_image)
    imageio.imwrite(case_dir / "final_selected_region.png", selected_image)

    payload = {
        "case": image_path.stem,
        "target_object": target,
        "goal": goal,
        "vlm_selected_object": selected_object,
        "vlm_selection_reason": selection_reason,
        "vlm_centroid_relative": list(relative_centroid),
        "vlm_centroid_pixel": list(centroid_xy),
        "grounding_prompt": grounding_prompt,
        "ranking": rankings,
        "selected_candidate_index": selected_index,
        "weights": {
            "confidence": CONFIDENCE_WEIGHT,
            "centroid": CENTROID_WEIGHT,
            "pixel_sigma": PIXEL_SIGMA,
        },
    }
    (case_dir / "result.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(f"Completed {image_path.name}: selected={selected_object!r}")
    print(f"  VLM selection reason: {selection_reason!r}")
    print(f"  GroundingDINO prompt: {grounding_prompt!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("localization_evaluation/scenes"))
    parser.add_argument("--output-dir", type=Path, default=Path("localization_evaluation/results"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for image_path in sorted(args.input_dir.glob("*.png")):
        run_case(image_path, args.output_dir)


if __name__ == "__main__":
    main()

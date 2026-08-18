#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from huggingface_hub import hf_hub_download

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
GROUNDING_DINO_ROOT = PROJECT_ROOT / "third_party" / "GroundingDINO"
sys.path.insert(0, str(GROUNDING_DINO_ROOT))

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.inference import predict
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict


def transform_image(image: Image.Image) -> torch.Tensor:
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ]
    )
    image_transformed, _ = transform(image, None)
    return image_transformed


def load_groundingdino(device: str):
    repo_id = "ShilongLiu/GroundingDINO"
    checkpoint_filename = "groundingdino_swinb_cogcoor.pth"
    config_filename = "GroundingDINO_SwinB.cfg.py"

    config_path = hf_hub_download(
        repo_id=repo_id,
        filename=config_filename,
    )
    checkpoint_path = hf_hub_download(
        repo_id=repo_id,
        filename=checkpoint_filename,
    )

    args = SLConfig.fromfile(config_path)
    args.device = device

    model = build_model(args)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )
    load_result = model.load_state_dict(
        clean_state_dict(checkpoint["model"]),
        strict=False,
    )

    model.to(device)
    model.eval()

    print(f"Loaded config: {config_path}")
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Model load result: {load_result}")

    return model


def draw_boxes(
    image: Image.Image,
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    phrases,
):
    result = image.copy()
    draw = ImageDraw.Draw(result)

    for box, score, phrase in zip(
        boxes_xyxy,
        scores,
        phrases,
    ):
        x1, y1, x2, y2 = [
            float(value) for value in box
        ]
        draw.rectangle(
            [x1, y1, x2, y2],
            outline="red",
            width=3,
        )
        draw.text(
            (x1, max(0.0, y1 - 14.0)),
            f"{phrase} {float(score):.3f}",
            fill="red",
        )

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run GroundingDINO only. SAM is not loaded."
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--visualization", required=True)
    parser.add_argument("--box-threshold", type=float, default=0.30)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    output_path = Path(args.output)
    visualization_path = Path(args.visualization)

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Input image does not exist: {image_path}"
        )

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is False."
        )

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform_image(image)

    model = load_groundingdino(args.device)

    boxes, scores, phrases = predict(
        model=model,
        image=image_tensor,
        caption=args.prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        device=args.device,
    )

    width, height = image.size

    boxes_xyxy = (
        box_ops.box_cxcywh_to_xyxy(boxes)
        * torch.tensor(
            [width, height, width, height],
            dtype=boxes.dtype,
            device=boxes.device,
        )
    )

    boxes_np = boxes_xyxy.detach().cpu().numpy()
    scores_np = scores.detach().cpu().numpy()
    phrases_np = np.asarray(list(phrases), dtype=str)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    visualization_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        boxes_xyxy=boxes_np,
        scores=scores_np,
        phrases=phrases_np,
        prompt=np.asarray(args.prompt),
        image_width=np.asarray(width),
        image_height=np.asarray(height),
    )

    visualization = draw_boxes(
        image,
        boxes_np,
        scores_np,
        phrases,
    )
    visualization.save(visualization_path)

    print(f"Input image: {image_path}")
    print(f"Prompt: {args.prompt}")
    print(f"Detection count: {len(boxes_np)}")

    for index, (box, score, phrase) in enumerate(
        zip(boxes_np, scores_np, phrases),
        start=1,
    ):
        print(
            f"{index}: phrase={phrase!r}, "
            f"score={float(score):.4f}, "
            f"box_xyxy={box.tolist()}"
        )

    print(f"Saved result: {output_path}")
    print(f"Saved visualization: {visualization_path}")


if __name__ == "__main__":
    main()

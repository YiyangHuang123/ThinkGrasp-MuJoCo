"""Qwen3-VL bridge for the MuJoCo ThinkGrasp closed loop.

This module uses a project-local frozen copy of the original ThinkGrasp
VLM system prompt (`vlm_system_prompt.txt`) and parses the same structured
response format that was validated in test_vlm_original_prompt.py.
"""

import base64
import os
from pathlib import Path

from openai import OpenAI
from PIL import Image


def load_image_as_base64(image_path):
    image_path = Path(image_path).expanduser().resolve()
    with open(image_path, "rb") as image_file:
        return base64.b64encode(
            image_file.read()
        ).decode("utf-8")


def load_system_prompt(system_prompt_path=None):
    """Load the project-local frozen copy of the original ThinkGrasp prompt."""

    if system_prompt_path is None:
        system_prompt_path = (
            Path(__file__).resolve().parent
            / "vlm_system_prompt.txt"
        )

    system_prompt_path = Path(
        system_prompt_path
    ).expanduser().resolve()

    if not system_prompt_path.is_file():
        raise FileNotFoundError(
            "VLM system prompt file does not exist: "
            f"{system_prompt_path}"
        )

    return system_prompt_path.read_text(
        encoding="utf-8"
    ).rstrip("\n")


def relative_coordinate_to_pixel(
    value,
    image_size,
):
    value = max(
        0,
        min(1000, int(value)),
    )

    return min(
        image_size - 1,
        int(
            round(
                value
                * image_size
                / 1000.0
            )
        ),
    )


def process_grasping_result(
    output,
    image_width,
    image_height,
):
    lines = output.strip().splitlines()

    result = {
        "selected_object": None,
        "cropping_box": None,
        "cropping_box_relative": None,
        "objects": [],
        "is_part": False,
        "raw_output": output,
    }

    for i, line in enumerate(lines):
        if line.startswith(
            "Selected Object/Object Part:"
        ):
            selected_object = line.split(
                ": ",
                1,
            )[1].strip()

            if selected_object.startswith(
                "[object part:"
            ):
                result["is_part"] = True
                selected_object = selected_object[
                    len("[object part:") :
                ].strip(" ]")
            else:
                result["is_part"] = False

                if selected_object.startswith(
                    "[object:"
                ):
                    selected_object = (
                        selected_object[
                            len("[object:") :
                        ].strip(" ]")
                    )

            result["selected_object"] = (
                selected_object
            )

        elif line.startswith(
            "Cropping Box Coordinates:"
        ):
            coords = line.split(
                ": ",
                1,
            )[1].strip()[1:-1]

            relative_box = tuple(
                int(value.strip())
                for value in coords.split(",")
            )

            if len(relative_box) != 4:
                raise ValueError(
                    "Expected four cropping-box "
                    "coordinates, got: "
                    f"{relative_box}"
                )

            result[
                "cropping_box_relative"
            ] = relative_box

            result["cropping_box"] = (
                relative_coordinate_to_pixel(
                    relative_box[0],
                    image_width,
                ),
                relative_coordinate_to_pixel(
                    relative_box[1],
                    image_height,
                ),
                relative_coordinate_to_pixel(
                    relative_box[2],
                    image_width,
                ),
                relative_coordinate_to_pixel(
                    relative_box[3],
                    image_height,
                ),
            )

        elif line.startswith("Object:"):
            if i + 6 >= len(lines):
                raise ValueError(
                    "Incomplete object block "
                    "in VLM response."
                )

            centroid_text = (
                lines[i + 5]
                .split(": ", 1)[1]
                .strip()[1:-1]
            )

            relative_centroid = tuple(
                int(value.strip())
                for value in centroid_text.split(",")
            )

            if len(relative_centroid) != 2:
                raise ValueError(
                    "Expected two centroid "
                    "coordinates, got: "
                    f"{relative_centroid}"
                )

            result["objects"].append(
                {
                    "name": (
                        line.split(
                            ": ",
                            1,
                        )[1].strip()
                    ),
                    "grasping_score": int(
                        lines[i + 1]
                        .split(": ", 1)[1]
                        .strip()
                    ),
                    "material_composition": int(
                        lines[i + 2]
                        .split(": ", 1)[1]
                        .strip()
                    ),
                    "surface_texture": int(
                        lines[i + 3]
                        .split(": ", 1)[1]
                        .strip()
                    ),
                    "stability_assessment": int(
                        lines[i + 4]
                        .split(": ", 1)[1]
                        .strip()
                    ),
                    "centroid_coordinates_relative": (
                        relative_centroid
                    ),
                    "centroid_coordinates": (
                        relative_coordinate_to_pixel(
                            relative_centroid[0],
                            image_width,
                        ),
                        relative_coordinate_to_pixel(
                            relative_centroid[1],
                            image_height,
                        ),
                    ),
                    "preferred_grasping_location": int(
                        lines[i + 6]
                        .split(": ", 1)[1]
                        .strip()
                    ),
                }
            )

    return result


def get_selected_object_properties(
    result,
):
    selected_object = result.get(
        "selected_object"
    )

    if not selected_object:
        raise ValueError(
            "VLM result does not contain "
            "a selected object."
        )

    selected_key = (
        selected_object
        .strip()
        .casefold()
        .replace("_", " ")
    )

    for obj in result.get("objects", []):
        object_key = (
            obj["name"]
            .strip()
            .casefold()
            .replace("_", " ")
        )

        if object_key == selected_key:
            return obj

    raise ValueError(
        "Could not match selected object "
        f"{selected_object!r} to any object "
        "property block in the VLM response."
    )


def run_vlm_selection(
    image_path,
    goal,
    system_prompt_path=None,
):
    image_path = Path(
        image_path
    ).expanduser().resolve()

    with Image.open(image_path) as image:
        image_width, image_height = image.size

    system_prompt = load_system_prompt(
        system_prompt_path
    )

    client = OpenAI(
        api_key=os.environ.get(
            "OPENAI_API_KEY",
            "EMPTY",
        ),
        base_url=os.environ.get(
            "VLM_BASE_URL",
            "http://127.0.0.1:8000/v1",
        ),
    )

    model = os.environ.get(
        "VLM_MODEL",
        "Qwen/Qwen3-VL-4B-Instruct",
    )

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": goal,
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": (
                            "data:image/png;base64,"
                            + load_image_as_base64(
                                image_path
                            )
                        )
                    },
                },
            ],
        },
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=713,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )

    raw_output = (
        response
        .choices[0]
        .message.content
    )

    result = process_grasping_result(
        raw_output,
        image_width=image_width,
        image_height=image_height,
    )

    selected_properties = (
        get_selected_object_properties(
            result
        )
    )

    preferred_location = int(
        selected_properties[
            "preferred_grasping_location"
        ]
    )

    if not 1 <= preferred_location <= 9:
        preferred_location = 5

    return {
        "result": result,
        "selected_object": result[
            "selected_object"
        ],
        "selected_properties": (
            selected_properties
        ),
        "preferred_grasping_location": (
            preferred_location
        ),
        "raw_output": raw_output,
    }

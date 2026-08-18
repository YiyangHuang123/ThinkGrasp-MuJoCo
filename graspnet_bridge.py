"""Subprocess bridge between MuJoCo and the legacy GraspNet environment."""

import os
from pathlib import Path
import subprocess

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_GRASPNET_PYTHON = os.environ.get(
    "GRASPNET_PYTHON"
)

GRASPNET_WORKER = (
    PROJECT_ROOT
    / "run_graspnet_inference.py"
)


def run_graspnet_inference(
    input_path,
    output_path,
    graspnet_python=None,
    cuda_visible_devices="0",
):
    """Run GraspNet in its separate environment.

    The caller remains in the MuJoCo environment. GraspNet is executed
    through a subprocess using the legacy ThinkGrasp Python interpreter.
    """

    input_path = Path(input_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()

    if graspnet_python is None:
        graspnet_python = DEFAULT_GRASPNET_PYTHON

    if not graspnet_python:
        raise RuntimeError(
            "GRASPNET_PYTHON is not set. "
            "Set it to the Python interpreter used for GraspNet."
        )

    graspnet_python = Path(
        graspnet_python
    ).expanduser().resolve()

    if not graspnet_python.exists():
        raise FileNotFoundError(
            "GraspNet Python interpreter does not exist: "
            f"{graspnet_python}\n"
            "Set the GRASPNET_PYTHON environment variable "
            "to the correct interpreter path."
        )

    if not GRASPNET_WORKER.exists():
        raise FileNotFoundError(
            f"GraspNet worker script does not exist: "
            f"{GRASPNET_WORKER}"
        )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input point-cloud file does not exist: "
            f"{input_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    command = [
        str(graspnet_python),
        str(GRASPNET_WORKER),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
    ]


    environment = os.environ.copy()

    if cuda_visible_devices is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(
            cuda_visible_devices
        )

    print("Running GraspNet subprocess:")
    print(" ".join(command))

    completed = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        env=environment,
        text=True,
        capture_output=True,
    )

    if completed.stdout:
        print(completed.stdout, end="")

    if completed.returncode != 0:
        if completed.stderr:
            print(completed.stderr, end="")

        raise RuntimeError(
            "GraspNet subprocess failed with return code "
            f"{completed.returncode}"
        )

    if not output_path.exists():
        raise RuntimeError(
            "GraspNet subprocess finished successfully, "
            f"but output file was not created: {output_path}"
        )

    return output_path


def load_target_grasps(output_path):
    """Load ranked grasps for the current GroundingDINO target crop."""

    output_path = Path(
        output_path
    ).expanduser().resolve()

    data = np.load(output_path)

    if "grasps" not in data.files:
        raise KeyError(
            f"{output_path} does not contain grasps"
        )

    return data["grasps"].astype(
        np.float64
    ).reshape(-1, 7)

def load_target_grasp_data(output_path):
    """Load ranked grasp poses together with GraspNet scores and angles.

    The arrays are aligned by index:
        grasps[i]
        grasp_scores[i]
        grasp_angles_deg[i]

    This is used by the paper-style final selection:
        nearest top-k by preferred-location XY distance
        -> highest GraspNet score within that top-k.
    """

    output_path = Path(
        output_path
    ).expanduser().resolve()

    data = np.load(output_path)

    required_keys = {
        "grasps",
        "grasp_scores",
        "grasp_angles_deg",
        "raw_graspnet_rotation_matrices",
        "converted_rotation_matrices",
        "diagnostic_grasp_centers",
        "diagnostic_source_indices",
    }

    missing_keys = required_keys.difference(data.files)

    if missing_keys:
        raise KeyError(
            f"{output_path} is missing keys: {sorted(missing_keys)}"
        )

    grasps = data["grasps"].astype(
        np.float64
    ).reshape(-1, 7)

    scores = data["grasp_scores"].astype(
        np.float64
    ).reshape(-1)

    angles = data["grasp_angles_deg"].astype(
        np.float64
    ).reshape(-1)

    if not (
        len(grasps)
        == len(scores)
        == len(angles)
    ):
        raise ValueError(
            "Grasp output arrays have inconsistent lengths: "
            f"grasps={len(grasps)}, "
            f"scores={len(scores)}, "
            f"angles={len(angles)}"
        )

    raw_rotations = data[
        "raw_graspnet_rotation_matrices"
    ].astype(np.float64).reshape(-1, 3, 3)

    converted_rotations = data[
        "converted_rotation_matrices"
    ].astype(np.float64).reshape(-1, 3, 3)

    diagnostic_centers = data[
        "diagnostic_grasp_centers"
    ].astype(np.float64).reshape(-1, 3)

    diagnostic_source_indices = data[
        "diagnostic_source_indices"
    ].astype(np.int64).reshape(-1)

    if not (
        len(grasps)
        == len(raw_rotations)
        == len(converted_rotations)
        == len(diagnostic_centers)
        == len(diagnostic_source_indices)
    ):
        raise ValueError(
            "Orientation diagnostic arrays are not aligned with grasps."
        )

    return {
        "grasps": grasps,
        "scores": scores,
        "angles_deg": angles,
        "raw_rotations": raw_rotations,
        "converted_rotations": converted_rotations,
        "diagnostic_centers": diagnostic_centers,
        "diagnostic_source_indices": diagnostic_source_indices,
    }


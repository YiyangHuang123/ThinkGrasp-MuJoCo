#!/usr/bin/env python3

import ast
import json
import os
import subprocess
import sys
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "graspnet"
    / "logs"
    / "log_rs"
    / "checkpoint.tar"
)

GROUNDEDINO_ROOT = (
    PROJECT_ROOT
    / "third_party"
    / "GroundingDINO"
)

NATIVE_RUNTIME = PROJECT_ROOT / ".native_runtime"

REQUIRED_ASSET_ROOT = (
    PROJECT_ROOT
    / "assets"
    / "scanned_objects"
    / "models"
)

ENVIRONMENT_SCRIPT = (
    PROJECT_ROOT
    / "thinkgrasp_minimal_env.py"
)


results = []


def record(name, ok, detail):
    results.append((name, ok, detail))
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}: {detail}")


def check_main_python():
    try:
        import mujoco
        import robosuite

        record(
            "MuJoCo main environment",
            True,
            f"python={sys.version.split()[0]}, "
            f"mujoco={mujoco.__version__}, "
            f"robosuite={robosuite.__version__}",
        )
    except Exception as exc:
        record(
            "MuJoCo main environment",
            False,
            repr(exc),
        )


def check_legacy_python():
    legacy = os.environ.get("LEGACY_PERCEPTION_PYTHON")

    if not legacy:
        record(
            "Legacy perception Python",
            False,
            "LEGACY_PERCEPTION_PYTHON is not set",
        )
        return None

    legacy_path = Path(legacy)

    if not legacy_path.is_file():
        record(
            "Legacy perception Python",
            False,
            f"not found: {legacy}",
        )
        return None

    try:
        output = subprocess.check_output(
            [
                legacy,
                "-c",
                (
                    "import sys, torch; "
                    "print(sys.version.split()[0]); "
                    "print(torch.__version__); "
                    "print(torch.cuda.is_available())"
                ),
            ],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip().splitlines()

        record(
            "Legacy perception Python",
            True,
            (
                f"python={output[0]}, "
                f"torch={output[1]}, "
                f"cuda_available={output[2]}"
            ),
        )
        return legacy

    except Exception as exc:
        record(
            "Legacy perception Python",
            False,
            repr(exc),
        )
        return None


def check_native_extensions(legacy):
    if legacy is None:
        record(
            "Native extensions",
            False,
            "skipped because legacy Python is unavailable",
        )
        return

    if not NATIVE_RUNTIME.is_dir():
        record(
            "Native extensions",
            False,
            f"missing runtime directory: {NATIVE_RUNTIME}",
        )
        return

    env = os.environ.copy()
    current_pythonpath = env.get("PYTHONPATH", "")

    path_entries = [
        str(NATIVE_RUNTIME),
        str(PROJECT_ROOT),
        str(GROUNDEDINO_ROOT),
    ]

    if current_pythonpath:
        path_entries.append(current_pythonpath)

    env["PYTHONPATH"] = os.pathsep.join(path_entries)

    code = r'''
import json
import torch
import pointnet22._ext
import knn_pytorch.knn_pytorch
import groundingdino._C

print(json.dumps({
    "pointnet22": pointnet22._ext.__file__,
    "knn": knn_pytorch.knn_pytorch.__file__,
    "groundingdino": groundingdino._C.__file__,
}))
'''

    try:
        output = subprocess.check_output(
            [legacy, "-c", code],
            text=True,
            stderr=subprocess.STDOUT,
            env=env,
        ).strip()

        info = json.loads(output)

        local_ok = (
            str(NATIVE_RUNTIME) in info["pointnet22"]
            and str(NATIVE_RUNTIME) in info["knn"]
            and str(GROUNDEDINO_ROOT) in info["groundingdino"]
        )

        record(
            "Native extensions",
            local_ok,
            (
                f"pointnet22={info['pointnet22']}; "
                f"knn={info['knn']}; "
                f"groundingdino={info['groundingdino']}"
            ),
        )

    except Exception as exc:
        record(
            "Native extensions",
            False,
            repr(exc),
        )


def check_checkpoint():
    record(
        "GraspNet checkpoint",
        CHECKPOINT.is_file(),
        str(CHECKPOINT.relative_to(PROJECT_ROOT)),
    )


def load_required_gso_models():
    """Read GSO model names directly from GSO_SCENE_OBJECT_SPECS.

    Parse the environment source with ast instead of importing it so the
    asset check stays independent of MuJoCo / robosuite initialization.
    """

    if not ENVIRONMENT_SCRIPT.is_file():
        raise FileNotFoundError(
            f"Missing environment script: {ENVIRONMENT_SCRIPT}"
        )

    source = ENVIRONMENT_SCRIPT.read_text(
        encoding="utf-8"
    )
    tree = ast.parse(
        source,
        filename=str(ENVIRONMENT_SCRIPT),
    )

    scene_specs = None

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue

        for target in node.targets:
            if (
                isinstance(target, ast.Name)
                and target.id == "GSO_SCENE_OBJECT_SPECS"
            ):
                scene_specs = ast.literal_eval(
                    node.value
                )
                break

        if scene_specs is not None:
            break

    if not isinstance(scene_specs, dict) or not scene_specs:
        raise RuntimeError(
            "Could not read GSO_SCENE_OBJECT_SPECS from "
            f"{ENVIRONMENT_SCRIPT.name}"
        )

    required_models = sorted(
        {
            object_spec["model_dir"]
            for scene_objects in scene_specs.values()
            for object_spec in scene_objects
        }
    )

    scene_object_slots = sum(
        len(scene_objects)
        for scene_objects in scene_specs.values()
    )

    return (
        scene_specs,
        required_models,
        scene_object_slots,
    )


def find_missing_model_files(model_dir):
    """Return files referenced by one GSO MJCF that are missing on disk."""

    model_xml = model_dir / "model.xml"

    if not model_xml.is_file():
        return [model_xml.name]

    missing = []

    try:
        root = ET.parse(
            model_xml
        ).getroot()
    except Exception as exc:
        return [
            f"model.xml (parse error: {exc})"
        ]

    if not (model_dir / "model.obj").is_file():
        missing.append("model.obj")

    for element in root.iter():
        file_attribute = element.get("file")

        if not file_attribute:
            continue

        referenced_path = (
            model_dir / file_attribute
        )

        if not referenced_path.is_file():
            missing.append(file_attribute)

    return sorted(set(missing))


def check_assets():
    try:
        (
            scene_specs,
            required_models,
            scene_object_slots,
        ) = load_required_gso_models()
    except Exception as exc:
        record(
            "Scanned-object assets",
            False,
            repr(exc),
        )
        return

    missing_models = []
    incomplete_models = []

    for name in required_models:
        model_dir = REQUIRED_ASSET_ROOT / name

        if not model_dir.is_dir():
            missing_models.append(name)
            continue

        missing_files = find_missing_model_files(
            model_dir
        )

        if missing_files:
            incomplete_models.append(
                (name, missing_files)
            )

    if missing_models or incomplete_models:
        details = []

        if missing_models:
            details.append(
                "missing model directories: "
                + ", ".join(missing_models)
            )

        if incomplete_models:
            incomplete_summary = "; ".join(
                (
                    f"{name}: "
                    + ", ".join(files)
                )
                for name, files
                in incomplete_models
            )
            details.append(
                "incomplete model files: "
                + incomplete_summary
            )

        record(
            "Scanned-object assets",
            False,
            " | ".join(details),
        )
        return

    record(
        "Scanned-object assets",
        True,
        (
            f"{len(scene_specs)} scenes, "
            f"{scene_object_slots} scene-object slots, "
            f"{len(required_models)} unique GSO models present"
        ),
    )


def check_vlm():
    base_url = os.environ.get(
        "VLM_BASE_URL",
        "http://127.0.0.1:8000/v1",
    ).rstrip("/")

    url = base_url + "/models"

    try:
        with urllib.request.urlopen(
            url,
            timeout=5,
        ) as response:
            payload = json.loads(
                response.read().decode("utf-8")
            )

        model_ids = [
            entry.get("id")
            for entry in payload.get("data", [])
        ]

        expected = os.environ.get(
            "VLM_MODEL",
            "Qwen/Qwen3-VL-4B-Instruct",
        )

        ok = expected in model_ids

        record(
            "Qwen vLLM endpoint",
            ok,
            f"url={url}, models={model_ids}",
        )

    except Exception as exc:
        record(
            "Qwen vLLM endpoint",
            False,
            f"url={url}, error={exc!r}",
        )


def main():
    print("ThinkGrasp-MuJoCo installation validation")
    print("Project root:", PROJECT_ROOT)
    print()

    check_main_python()
    legacy = check_legacy_python()
    check_native_extensions(legacy)
    check_checkpoint()
    check_assets()
    check_vlm()

    print()

    failed = [
        name
        for name, ok, _ in results
        if not ok
    ]

    if failed:
        print("Validation result: FAIL")
        print("Failed checks:")
        for name in failed:
            print(" -", name)
        raise SystemExit(1)

    print("Validation result: PASS")


if __name__ == "__main__":
    main()

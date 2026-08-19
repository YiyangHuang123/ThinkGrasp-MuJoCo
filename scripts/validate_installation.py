#!/usr/bin/env python3

import importlib.util
import json
import os
import subprocess
import sys
import urllib.request
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

REQUIRED_OBJECTS = [
    "ACE_Coffee_Mug_Kristen_16_oz_cup",
    "BIA_Porcelain_Ramekin_With_Glazed_Rim_35_45_oz_cup",
    "Canon_Pixma_Ink_Cartridge_8",
    "Circo_Fish_Toothbrush_Holder_14995988",
    "Ecoforms_Cup_B4_SAN",
]


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


def check_assets():
    missing = []

    for name in REQUIRED_OBJECTS:
        model_dir = REQUIRED_ASSET_ROOT / name

        if not (model_dir / "model.obj").is_file():
            missing.append(name)

    if missing:
        record(
            "Scanned-object assets",
            False,
            "missing: " + ", ".join(missing),
        )
    else:
        record(
            "Scanned-object assets",
            True,
            f"{len(REQUIRED_OBJECTS)} required objects present",
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

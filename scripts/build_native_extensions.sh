#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="$PROJECT_ROOT/.native_runtime"

: "${LEGACY_PERCEPTION_PYTHON:?Set LEGACY_PERCEPTION_PYTHON to the Python 3.8 perception environment}"
: "${CUDA_HOME:?Set CUDA_HOME to a CUDA toolkit compatible with the legacy PyTorch environment}"

if [[ ! -x "$LEGACY_PERCEPTION_PYTHON" ]]; then
    echo "ERROR: LEGACY_PERCEPTION_PYTHON is not executable:"
    echo "  $LEGACY_PERCEPTION_PYTHON"
    exit 1
fi

if [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
    echo "ERROR: nvcc not found at:"
    echo "  $CUDA_HOME/bin/nvcc"
    exit 1
fi

export PATH="$CUDA_HOME/bin:$PATH"

rm -rf "$RUNTIME_DIR"
mkdir -p "$RUNTIME_DIR"

echo "=== Building GroundingDINO CUDA extension ==="

cd "$PROJECT_ROOT/third_party/GroundingDINO"

rm -rf build
find groundingdino -type f -name '_C*.so' -delete

"$LEGACY_PERCEPTION_PYTHON" setup.py build_ext --inplace

echo "=== Building PointNet2 CUDA extension ==="

cd "$PROJECT_ROOT/models/graspnet/pointnet2"

rm -rf build
"$LEGACY_PERCEPTION_PYTHON" setup.py build_ext

POINTNET_BUILD_ROOT="$(
    find build -maxdepth 1 -type d -name 'lib.*' -print -quit
)"

if [[ -z "$POINTNET_BUILD_ROOT" ]]; then
    echo "ERROR: PointNet2 build/lib.* directory not found"
    exit 1
fi

cp -a "$POINTNET_BUILD_ROOT/pointnet22" "$RUNTIME_DIR/"

echo "=== Building KNN CUDA extension ==="

cd "$PROJECT_ROOT/models/graspnet/knn"

rm -rf build
"$LEGACY_PERCEPTION_PYTHON" setup.py build_ext

KNN_BUILD_ROOT="$(
    find build -maxdepth 1 -type d -name 'lib.*' -print -quit
)"

if [[ -z "$KNN_BUILD_ROOT" ]]; then
    echo "ERROR: KNN build/lib.* directory not found"
    exit 1
fi

cp -a "$KNN_BUILD_ROOT/knn_pytorch" "$RUNTIME_DIR/"

echo "=== Native extension smoke test ==="

cd "$PROJECT_ROOT"

PYTHONPATH="$RUNTIME_DIR:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
"$LEGACY_PERCEPTION_PYTHON" - <<'PY'
import torch
import pointnet22._ext
import knn_pytorch.knn_pytorch

print("torch:", torch.__version__)
print("pointnet22:", pointnet22._ext.__file__)
print("knn:", knn_pytorch.knn_pytorch.__file__)
print("Native extension smoke test: OK")
PY

echo "Native extensions successfully built."
echo "Runtime directory:"
echo "  $RUNTIME_DIR"

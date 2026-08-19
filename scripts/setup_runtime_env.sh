#!/usr/bin/env bash

# Source this file:
#   source scripts/setup_runtime_env.sh
#
# Machine-specific paths such as Python environments must be supplied by
# the user before sourcing this script.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NATIVE_RUNTIME="$PROJECT_ROOT/.native_runtime"

export PROJECT_ROOT

export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"

export VLM_BASE_URL="${VLM_BASE_URL:-http://127.0.0.1:8000/v1}"
export VLM_MODEL="${VLM_MODEL:-Qwen/Qwen3-VL-4B-Instruct}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

if [[ -d "$NATIVE_RUNTIME" ]]; then
    export PYTHONPATH="$NATIVE_RUNTIME:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
else
    echo "WARNING: project-local native runtime not found:"
    echo "  $NATIVE_RUNTIME"
    echo "Run scripts/build_native_extensions.sh first."
fi

if [[ -z "${LEGACY_PERCEPTION_PYTHON:-}" ]]; then
    echo "WARNING: LEGACY_PERCEPTION_PYTHON is not set."
fi

if [[ -z "${GRASPNET_PYTHON:-}" ]]; then
    if [[ -n "${LEGACY_PERCEPTION_PYTHON:-}" ]]; then
        export GRASPNET_PYTHON="$LEGACY_PERCEPTION_PYTHON"
    else
        echo "WARNING: GRASPNET_PYTHON is not set."
    fi
fi

echo "ThinkGrasp-MuJoCo runtime environment configured."
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "MUJOCO_GL=$MUJOCO_GL"
echo "PYOPENGL_PLATFORM=$PYOPENGL_PLATFORM"
echo "VLM_BASE_URL=$VLM_BASE_URL"
echo "VLM_MODEL=$VLM_MODEL"

if [[ -n "${LEGACY_PERCEPTION_PYTHON:-}" ]]; then
    echo "LEGACY_PERCEPTION_PYTHON=$LEGACY_PERCEPTION_PYTHON"
fi

if [[ -n "${GRASPNET_PYTHON:-}" ]]; then
    echo "GRASPNET_PYTHON=$GRASPNET_PYTHON"
fi

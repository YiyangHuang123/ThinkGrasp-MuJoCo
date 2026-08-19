# ThinkGrasp-MuJoCo

A MuJoCo migration and closed-loop reproduction of the ThinkGrasp robotic grasping pipeline.

This project replaces the original PyBullet simulation environment with MuJoCo while preserving the main ThinkGrasp perception and grasp-planning logic. The current simulator uses a Franka Emika Panda with the Franka Hand.

## Pipeline

The current closed-loop pipeline is:

```text
RGB image
    ↓
VLM target selection
(Qwen3-VL-4B-Instruct)
    ↓
GroundingDINO target localization
    ↓
MuJoCo RGB-D / point-cloud reconstruction
    ↓
GraspNet grasp proposal generation
    ↓
Simulator-object assignment and source-style filtering
    ↓
Preferred grasp selection
    ↓
Panda IK
    ↓
JOINT_POSITION control
    ↓
Grasp → lift → transport → release
    ↓
Simulator-side success evaluation
    ↓
Retry if necessary
```

The formal grasp-control path is:

```text
GraspNet 7D grasp pose
→ Panda grip-site target pose
→ inverse kinematics
→ 7 Panda joint targets
→ joint-position control
```

OSC is not used as the formal grasp execution controller.

## Repository Structure

```text
.
├── run_closed_loop.py
├── thinkgrasp_minimal_env.py
├── scene_bridge.py
├── vlm_bridge.py
├── graspnet_bridge.py
├── graspnet_config.py
├── grasp_detetor.py
├── run_graspnet_inference.py
├── run_groundingdino_inference.py
├── run_pointcloud_fusion.py
├── pybullet_pointcloud_fusion.py
├── perception_viz.py
├── dual_view_recorder.py
├── vlm_system_prompt.txt
│
├── cases/
├── assets/
│   └── scanned_objects/
├── models/
│   └── graspnet/
├── scripts/
│   ├── build_native_extensions.sh
│   ├── setup_runtime_env.sh
│   └── validate_installation.py
└── third_party/
    └── GroundingDINO/
```

Runtime outputs, videos, logs, compiled extensions, model checkpoints, development archives, and local milestone snapshots are excluded from Git.

## Python Environments

The project currently uses two Python environments.

### 1. MuJoCo main environment

Validated environment:

```text
Python      3.10.20
NumPy       2.2.6
SciPy       1.15.3
MuJoCo      3.3.7
robosuite   1.5.2
openai      2.53.0
```

This environment runs:

- MuJoCo / robosuite simulation
- Panda control
- scene generation
- VLM communication
- closed-loop execution

Open3D is not required in the main MuJoCo environment.

### 2. Legacy perception / GraspNet environment

Validated environment:

```text
Python          3.8.20
NumPy           1.23.5
PyTorch         1.13.1+cu117
torchvision     0.14.1+cu117
Open3D          0.15.2
Transformers    4.46.3
timm            1.0.28
huggingface_hub 0.36.2
```

This environment is used through subprocesses for:

- GroundingDINO
- source-style point-cloud fusion
- GraspNet

The main MuJoCo environment and the legacy perception environment are intentionally kept separate.

## Required Environment Variables

Before building native extensions or running the closed loop, define the Python interpreter used by the legacy perception workers:

```bash
export LEGACY_PERCEPTION_PYTHON=/path/to/legacy/python3.8
```

GraspNet can use the same interpreter:

```bash
export GRASPNET_PYTHON="$LEGACY_PERCEPTION_PYTHON"
```

If a separate GraspNet environment is used instead:

```bash
export GRASPNET_PYTHON=/path/to/graspnet/python
```

For native extension compilation, also define a CUDA toolkit compatible with the PyTorch version installed in the legacy perception environment:

```bash
export CUDA_HOME=/path/to/compatible/cuda/toolkit
```

Do not assume that the system-wide CUDA toolkit is compatible with the legacy PyTorch build.

## Native CUDA Extensions

GroundingDINO, PointNet2, and KNN require compiled native extensions.

Build all required extensions with:

```bash
bash scripts/build_native_extensions.sh
```

The script builds:

```text
GroundingDINO CUDA extension
PointNet2 CUDA extension
KNN CUDA extension
```

The PointNet2 and KNN runtime packages are copied into:

```text
.native_runtime/
```

This directory is generated locally and excluded from Git.

GroundingDINO is built in place under:

```text
third_party/GroundingDINO/
```

Normal project execution does not require temporary `/tmp` extension directories.

## VLM

The current VLM is:

```text
Qwen/Qwen3-VL-4B-Instruct
```

`vlm_bridge.py` communicates with an OpenAI-compatible API endpoint.

The Qwen model is served separately from the MuJoCo process. A validated deployment uses vLLM.

Prepare a vLLM environment and a local Qwen model directory:

```bash
export QWEN_ENV=/path/to/qwen_vllm_environment
export QWEN_MODEL_PATH=/path/to/Qwen3-VL-4B-Instruct
```

Start the Qwen service in a dedicated terminal:

```bash
CUDA_VISIBLE_DEVICES=1 \
"$QWEN_ENV/bin/vllm" serve \
"$QWEN_MODEL_PATH" \
  --served-model-name Qwen/Qwen3-VL-4B-Instruct \
  --host 127.0.0.1 \
  --port 8000 \
  --gpu-memory-utilization 0.75 \
  --max-model-len 4096 \
  --dtype bfloat16
```

The GPU index and memory utilization may be adjusted for the target machine.

In the terminal used to run ThinkGrasp-MuJoCo, configure the client:

```bash
export VLM_BASE_URL=http://127.0.0.1:8000/v1
export VLM_MODEL=Qwen/Qwen3-VL-4B-Instruct
export OPENAI_API_KEY=EMPTY
```

Verify that the service is available before starting the closed loop:

```bash
curl -s http://127.0.0.1:8000/v1/models
```

The response should list:

```text
Qwen/Qwen3-VL-4B-Instruct
```

The ThinkGrasp-style VLM system prompt used by this project is stored locally in:

```text
vlm_system_prompt.txt
```

This avoids a runtime dependency on the original ThinkGrasp `simulation_main.py`.

## GroundingDINO

GroundingDINO source code is stored locally under:

```text
third_party/GroundingDINO/
```

The project therefore does not require the original parent ThinkGrasp repository at runtime.

`run_groundingdino_inference.py` uses the project-local GroundingDINO source.

Current model files requested by the worker are:

```text
GroundingDINO_SwinB.cfg.py
groundingdino_swinb_cogcoor.pth
```

The GroundingDINO native extension is built by:

```bash
bash scripts/build_native_extensions.sh
```

## GraspNet

The project-local GraspNet source is stored under:

```text
models/graspnet/
```

The GraspNet checkpoint is intentionally excluded from Git.

Place the checkpoint at:

```text
models/graspnet/logs/log_rs/checkpoint.tar
```

The current project configuration is stored in:

```text
graspnet_config.py
```

The formal closed loop uses GraspNet point-cloud inference without semantic text input. Target association and source-style grasp filtering are performed by the MuJoCo pipeline after grasp generation.

PointNet2 and KNN extensions are built by:

```bash
bash scripts/build_native_extensions.sh
```

Their project-local runtime copies are stored under:

```text
.native_runtime/
```

## MuJoCo Assets

The formal scene currently uses five scanned objects:

```text
ACE_Coffee_Mug_Kristen_16_oz_cup
Ecoforms_Cup_B4_SAN
Circo_Fish_Toothbrush_Holder_14995988
BIA_Porcelain_Ramekin_With_Glazed_Rim_35_45_oz_cup
Canon_Pixma_Ink_Cartridge_8
```

Only these required models and their robosuite adapters are intended to be tracked by this repository.

The original scanned-object dataset metadata and license files are retained under:

```text
assets/scanned_objects/
```

## Headless MuJoCo

For a headless server using OSMesa:

```bash
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
```

These values are also provided as defaults by:

```bash
source scripts/setup_runtime_env.sh
```

## Runtime Environment Setup

After the native extensions have been built, configure the project runtime environment.

First define machine-specific interpreter paths:

```bash
export LEGACY_PERCEPTION_PYTHON=/path/to/legacy/python3.8
export GRASPNET_PYTHON="$LEGACY_PERCEPTION_PYTHON"
```

Then source the project setup script:

```bash
source scripts/setup_runtime_env.sh
```

The script configures:

```text
PROJECT_ROOT
PYTHONPATH
MUJOCO_GL
PYOPENGL_PLATFORM
VLM_BASE_URL
VLM_MODEL
OPENAI_API_KEY
GRASPNET_PYTHON
```

Machine-specific absolute paths are intentionally not stored in the repository.

## Installation Validation

Before running the full closed loop, validate the installation:

```bash
python scripts/validate_installation.py
```

The validator checks:

```text
MuJoCo main environment
Legacy perception Python / PyTorch / CUDA availability
Project-local PointNet2 extension
Project-local KNN extension
Project-local GroundingDINO extension
GraspNet checkpoint
Required scanned-object assets
Qwen vLLM endpoint
```

A fully configured installation should report:

```text
[PASS] MuJoCo main environment
[PASS] Legacy perception Python
[PASS] Native extensions
[PASS] GraspNet checkpoint
[PASS] Scanned-object assets
[PASS] Qwen vLLM endpoint

Validation result: PASS
```

A robosuite warning about the optional Mink-based whole-body IK controller for GR1 may appear during import. The current Panda pipeline uses its own IK and joint-position control path and does not rely on that GR1 controller.

## Running the Closed Loop

A minimal validated workflow is:

```bash
# 1. Activate the MuJoCo environment.
conda activate /path/to/thinkgrasp_mujoco

# 2. Define machine-specific legacy environment paths.
export LEGACY_PERCEPTION_PYTHON=/path/to/legacy/python3.8
export GRASPNET_PYTHON="$LEGACY_PERCEPTION_PYTHON"

# 3. Configure the project runtime.
source scripts/setup_runtime_env.sh

# 4. Verify the installation and Qwen service.
python scripts/validate_installation.py

# 5. Run the closed loop.
python run_closed_loop.py
```

The Qwen vLLM service must already be running in a separate terminal before the validator or closed-loop process is started.

The current default testcase is:

```text
cases/case02_white_ramekin.txt
```

A different testcase can be selected with:

```bash
python run_closed_loop.py --case cases/case01_red_mug.txt
```

A case file contains two lines:

```text
natural-language grasp instruction
MuJoCo ground-truth object name
```

For example:

```text
pick up the white ramekin
white_ramekin
```

The language instruction is used by the VLM / GroundingDINO / grasp-planning pipeline.

The simulator ground-truth object name is used only for final task-success evaluation and is not used to pre-filter VLM, GroundingDINO, or GraspNet predictions.

## Recommended Setup Order

For a clean clone, the recommended order is:

```text
Clone repository
    ↓
Create / activate MuJoCo main environment
    ↓
Prepare legacy perception / GraspNet environment
    ↓
Set LEGACY_PERCEPTION_PYTHON and CUDA_HOME
    ↓
Build native extensions
    ↓
Place GraspNet checkpoint
    ↓
Prepare and start Qwen3-VL with vLLM
    ↓
Source scripts/setup_runtime_env.sh
    ↓
Run scripts/validate_installation.py
    ↓
Run run_closed_loop.py
```

## Runtime Outputs

Runtime artifacts are generated under directories such as:

```text
bridge_data/
closed_loop_outputs/
grasp_videos/
```

Generated point clouds, NPZ files, PLY visualizations, videos, logs, compiled extensions, and other runtime artifacts are excluded from Git.

## Third-Party Components

This repository contains project-local copies or subsets of third-party components used by the original ThinkGrasp pipeline.

Please refer to the corresponding license and README files:

```text
models/graspnet/LICENSE
models/graspnet/README.md

third_party/GroundingDINO/LICENSE
third_party/GroundingDINO/README.md

assets/scanned_objects/LICENSE
assets/scanned_objects/README.md
assets/scanned_objects/VERSION
```

These components remain subject to their respective upstream licenses.

## Project Status

Current validated components include:

- MuJoCo Panda scene
- RGB-D perception
- four-view point-cloud fusion
- VLM target selection
- GroundingDINO localization
- GraspNet inference
- source-style grasp filtering
- Panda inverse kinematics
- joint-position grasp execution
- gripper closing
- lift and transport
- simulator-side grasp-success evaluation
- retry-based closed-loop execution
- grasp and perception debugging outputs
- project-local native extension build workflow
- clean-clone standalone validation workflow

A clean-clone standalone validation has confirmed that the main perception and execution chain can run without importing code from the original parent ThinkGrasp repository.

The project is under active development as part of a bachelor thesis on VLM-based robotic grasping and simulation migration.

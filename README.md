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

Before running the closed loop, define the Python interpreters used by the subprocess workers.

```bash
export LEGACY_PERCEPTION_PYTHON=/path/to/legacy_python3.8
export GRASPNET_PYTHON=/path/to/graspnet_python
```

The VLM client uses:

```bash
export VLM_BASE_URL=http://127.0.0.1:8000/v1
export VLM_MODEL=Qwen/Qwen3-VL-4B-Instruct
```

If the endpoint requires an API key:

```bash
export OPENAI_API_KEY=your_api_key
```

For a local OpenAI-compatible endpoint that does not require authentication, the code defaults to:

```text
OPENAI_API_KEY=EMPTY
```

## VLM

The current VLM is:

```text
Qwen/Qwen3-VL-4B-Instruct
```

`vlm_bridge.py` communicates with an OpenAI-compatible API endpoint.

The default endpoint is:

```text
http://127.0.0.1:8000/v1
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

`run_groundingdino_inference.py` downloads the GroundingDINO configuration and checkpoint through `huggingface_hub` when required.

Current model files requested by the worker:

```text
GroundingDINO_SwinB.cfg.py
groundingdino_swinb_cogcoor.pth
```

Compiled GroundingDINO extensions are not committed to Git and must be built or installed for the target environment.

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

Compiled PointNet2 / KNN extensions and build artifacts are not committed to Git and must be built in the GraspNet environment.

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

For the current headless server configuration:

```bash
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
```

Then activate the MuJoCo environment.

## Running the Closed Loop

Example:

```bash
conda activate /path/to/thinkgrasp_mujoco

export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

export LEGACY_PERCEPTION_PYTHON=/path/to/legacy_python3.8
export GRASPNET_PYTHON=/path/to/graspnet_python

export VLM_BASE_URL=http://127.0.0.1:8000/v1
export VLM_MODEL=Qwen/Qwen3-VL-4B-Instruct

python run_closed_loop.py
```

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

## Runtime Outputs

Runtime artifacts are generated under directories such as:

```text
bridge_data/
closed_loop_outputs/
grasp_videos/
```

Generated point clouds, NPZ files, PLY visualizations, videos, and logs are excluded from Git.

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

The project is under active development as part of a bachelor thesis on VLM-based robotic grasping and simulation migration.

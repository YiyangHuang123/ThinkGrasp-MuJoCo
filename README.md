# ThinkGrasp-MuJoCo

A MuJoCo migration and closed-loop reproduction of the ThinkGrasp robotic grasping pipeline.

This project replaces the original PyBullet simulation environment with MuJoCo while preserving the main ThinkGrasp perception and grasp-planning ideas. The current simulator uses a Franka Emika Panda with the Franka Hand.

## Pipeline

The current formal closed-loop pipeline is:

```text
Natural-language task
    ↓
MuJoCo RGB-D perception
    ↓
Qwen3-VL target selection
    ↓
GroundingDINO target localization
    ↓
GraspNet grasp proposal generation
    ↓
GroundingDINO target-region grasp filtering
    ↓
weighted grasp ranking
(angle quality + VLM preferred location)
    ↓
Panda inverse kinematics
    ↓
JOINT_POSITION / q_ref execution
    ↓
grasp → lift → fixed joint-space transport → release
    ↓
simulator-side task evaluation
    ↓
re-perception / re-planning if necessary
```

If the selected target region contains no usable grasp, the runner performs a four-view full-scene fallback. The fallback is intentionally target-agnostic and is used only to perturb the clutter before the next perception cycle.

The formal grasp-control path is:

```text
GraspNet 7D grasp pose
→ Panda grip-site target pose
→ inverse kinematics
→ 7 Panda joint targets
→ JOINT_POSITION / q_ref control
```

OSC compatibility utilities remain in `thinkgrasp_minimal_env.py`, but OSC is not used by the formal closed-loop grasp execution path.

## Current Grasp-Selection Policy

GroundingDINO detections are ranked with a soft combination of detector confidence and VLM centroid proximity:

```text
0.70 × GroundingDINO confidence
+ 0.30 × VLM-centroid proximity score
```

Inside the selected GroundingDINO target region, the normal grasp selector uses:

```text
0.60 × approach-angle score
+ 0.40 × preferred-location score
```

No hard approach-angle threshold is applied in normal target-grasp selection. GraspNet confidence is diagnostic in this mode.

When the target region contains zero usable grasps, the full-scene fallback uses:

```text
0.60 × approach-angle score
+ 0.40 × GraspNet confidence
```

The VLM preferred location is intentionally ignored in fallback mode.

## Repository Structure

```text
.
├── run_closed_loop.py
├── thinkgrasp_minimal_env.py
├── scene_bridge.py
├── vlm_bridge.py
├── graspnet_bridge.py
├── graspnet_config.py
├── grasp_detector.py
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
│       ├── models/
│       └── robosuite_adapters/        # generated locally
├── models/
│   └── graspnet/
├── scripts/
│   ├── build_native_extensions.sh
│   ├── setup_runtime_env.sh
│   └── validate_installation.py
└── third_party/
    └── GroundingDINO/
```

Runtime outputs, videos, logs, compiled extensions, model checkpoints, development archives, local milestone snapshots, and generated robosuite adapter XML files are excluded from Git.

## Python Environments

The project uses two execution environments, plus a separate environment for serving the VLM when vLLM is used.

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
- VLM API communication
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

The MuJoCo process and legacy perception workers are intentionally kept separate.

## Required Environment Variables

Before building native extensions or running the closed loop, define the Python interpreter used by the legacy perception workers:

```bash
export LEGACY_PERCEPTION_PYTHON=/path/to/legacy/python3.8
```

GraspNet can use the same interpreter:

```bash
export GRASPNET_PYTHON="$LEGACY_PERCEPTION_PYTHON"
```

If a separate GraspNet environment is used:

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

Build them with:

```bash
bash scripts/build_native_extensions.sh
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

## VLM

The current VLM is:

```text
Qwen/Qwen3-VL-4B-Instruct
```

`vlm_bridge.py` communicates with an OpenAI-compatible API endpoint.

A validated deployment uses vLLM. Prepare a vLLM environment and local model directory:

```bash
export QWEN_ENV=/path/to/qwen_vllm_environment
export QWEN_MODEL_PATH=/path/to/Qwen3-VL-4B-Instruct
```

Start the service in a dedicated terminal:

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

Configure the ThinkGrasp-MuJoCo client:

```bash
export VLM_BASE_URL=http://127.0.0.1:8000/v1
export VLM_MODEL=Qwen/Qwen3-VL-4B-Instruct
export OPENAI_API_KEY=EMPTY
```

Verify the endpoint:

```bash
curl -s http://127.0.0.1:8000/v1/models
```

The ThinkGrasp-style system prompt is stored locally in:

```text
vlm_system_prompt.txt
```

## GroundingDINO

GroundingDINO source code is stored under:

```text
third_party/GroundingDINO/
```

`run_groundingdino_inference.py` therefore does not require the original parent ThinkGrasp repository at runtime.

The current worker requests:

```text
GroundingDINO_SwinB.cfg.py
groundingdino_swinb_cogcoor.pth
```

GroundingDINO receives a high-resolution RAW top-view crop derived from the configured world workspace. The crop is computed from RAW per-pixel world coordinates rather than from a fixed image-space rectangle.

## GraspNet

The project-local GraspNet source is stored under:

```text
models/graspnet/
```

The checkpoint is intentionally excluded from Git. Place it at:

```text
models/graspnet/logs/log_rs/checkpoint.tar
```

The project configuration is stored in:

```text
graspnet_config.py
```

GraspNet itself receives point clouds without semantic text input. Target-specific filtering and final ranking are performed after grasp generation by the MuJoCo closed-loop runner.

## MuJoCo Scenes and GSO Assets

The evaluation setup contains 10 fixed five-object clutter scenes. Together they use 50 scene-object slots and 49 unique Google Scanned Object model directories because `BUNNY_RACER` is intentionally reused in Scene02 and Scene09.

The robot, table, cameras, workspace, clutter-drop procedure, bin, controller, reward logic, and closed-loop task logic remain shared across the scenes; only the selected five-object set changes.

| Scene | Object aliases | Fixed default target |
| --- | --- | --- |
| scene01 | coffee_mug, ecoforms_cup, circo_holder, **white_ramekin**, ink_cartridge | `white_ramekin` |
| scene02 | black_bowl, nesquik_canister, crayon_box, **nikon_camera**, bunny_racer | `nikon_camera` |
| scene03 | white_cereal_bowl, latte_box, green_speaker, **mario_figure**, can_opener | `mario_figure` |
| scene04 | turquoise_bowl, fondant_box, blue_bottle, **baby_car**, alarm_clock | `baby_car` |
| scene05 | yellow_blue_bowl, mocha_box, **gaming_mouse**, yoshi_figure, black_ink_box | `gaming_mouse` |
| scene06 | quercetin_bottle, cookie_candy_box, fire_truck, **moisturizer_jar**, pencil_case | `moisturizer_jar` |
| scene07 | probiotic_bottle, snack_dispenser, **rhino_figure**, color_ink_box, hard_drive | `rhino_figure` |
| scene08 | **creatine_bottle**, fujifilm_camera_box, speed_boat, face_moisturizer, peanut_butter_candy_box | `creatine_bottle` |
| scene09 | neck_cream_jar, **lion_figure**, soap_dish, pink_rubber_toy, bunny_racer | `lion_figure` |
| scene10 | borage_bottle, toy_airplane, game_case, **crocodile_toy**, cleanser_bottle | `crocodile_toy` |

The complete alias-to-GSO-directory mapping is defined in:

```text
GSO_SCENE_OBJECT_SPECS
```

inside `thinkgrasp_minimal_env.py`.

Only the GSO source model directories referenced by these 10 scenes are intended to be tracked by this repository. The remaining scanned-object dataset is ignored by `.gitignore`.

At runtime, `thinkgrasp_minimal_env.py` generates robosuite-compatible adapter XML files under:

```text
assets/scanned_objects/robosuite_adapters/
```

These adapters are derived from the tracked GSO source models and are excluded from Git.

The original scanned-object dataset metadata and license files remain under:

```text
assets/scanned_objects/
```

## Headless MuJoCo

For a headless server using OSMesa:

```bash
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
```

These are also the defaults used by:

```bash
source scripts/setup_runtime_env.sh
```

## Runtime Environment Setup

After the native extensions have been built, configure the runtime environment.

First define machine-specific interpreter paths:

```bash
export LEGACY_PERCEPTION_PYTHON=/path/to/legacy/python3.8
export GRASPNET_PYTHON="$LEGACY_PERCEPTION_PYTHON"
```

Then source:

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

The validator is intended to catch missing core runtime dependencies before a long closed-loop run. Scene-specific GSO source assets are also checked when the selected MuJoCo scene is constructed.

A robosuite warning about the optional Mink-based whole-body IK controller for GR1 may appear during import. The current Panda pipeline uses its own IK and JOINT_POSITION path and does not rely on that GR1 controller.

## Running the Closed Loop

The normal interface selects a scene number. The runner automatically resolves that scene to its fixed default case:

```bash
python run_closed_loop.py --scene 1
```

For example:

```bash
python run_closed_loop.py --scene 5
```

resolves to:

```text
scene05
→ cases/case_scene05_gaming_mouse.txt
→ target: gaming_mouse
```

Both numeric and canonical scene names are accepted:

```bash
python run_closed_loop.py --scene 5
python run_closed_loop.py --scene scene05
```

If `--scene` is omitted, Scene01 is used.

A case can still be supplied explicitly as a manual override:

```bash
python run_closed_loop.py \
  --scene 5 \
  --case cases/case_scene05_gaming_mouse.txt
```

A case file contains at least two lines:

```text
natural-language grasp instruction
MuJoCo ground-truth target object name
```

For example:

```text
Pick up the white ramekin and place it in the bin.
white_ramekin
```

The natural-language instruction is used by the VLM / GroundingDINO / grasp-planning pipeline.

The simulator ground-truth target name is not used to pre-filter VLM, GroundingDINO, or GraspNet predictions. It is used for simulator-side task evaluation and reward bookkeeping.

A minimal workflow is:

```bash
# 1. Activate the MuJoCo environment.
conda activate /path/to/thinkgrasp_mujoco

# 2. Define the legacy perception worker.
export LEGACY_PERCEPTION_PYTHON=/path/to/legacy/python3.8
export GRASPNET_PYTHON="$LEGACY_PERCEPTION_PYTHON"

# 3. Configure project-local runtime paths and defaults.
source scripts/setup_runtime_env.sh

# 4. Validate the installation and VLM endpoint.
python scripts/validate_installation.py

# 5. Run one fixed evaluation scene.
python run_closed_loop.py --scene 1
```

The Qwen vLLM service must already be running in a separate terminal.

## Formal Execution Behavior

The current closed-loop execution uses:

```text
pregrasp height:        0.20 m in world +Z
lift height:            0.20 m in world +Z
grasp depth offset:     0.0 m
Cartesian waypoint:     0.01 m
q_ref speed:            1.0 rad/s
force-stop threshold:   15.0
force-stop persistence: 5 consecutive physics steps
```

IK uses continuity-first seed selection: the current / previous waypoint joint state is preferred whenever it is already practically acceptable, and deterministic multi-start solutions are used only as fallback.

The gripper closes until its width becomes stable, with an 80-control-step maximum.

If lift IK fails after the gripper has already closed, the object is released at the current pose before the Panda returns home. Failures before lift recover home with the gripper open.

After a successful lift, the held object is transported with a fixed Panda joint-space drop posture rather than a separate bin-target IK stage.

## Reward and Task Evaluation

The runner keeps physical grasp success, reward, and final task success as separate concepts.

Current reward bookkeeping follows the original ThinkGrasp-style semantics:

```text
empty grasp / execution or transport failure:  -1
correct target grasp and transport:             +2
wrong object grasp:
    - distance(actual grasped object, target) / workspace XY diagonal
```

After lift, the actually grasped object is inferred from simulator object state. Final task completion is evaluated after release and return-home by checking whether the ground-truth target body lies inside the receiving-bin XY footprint.

The accumulated value is printed as:

```text
Final episode reward
```

## Recommended Setup Order

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
Run run_closed_loop.py --scene <1..10>
```

## Runtime Outputs

Runtime artifacts are generated under directories such as:

```text
bridge_data/
closed_loop_outputs/
grasp_videos/
```

`closed_loop_outputs/` contains human-readable debugging artifacts such as VLM-selection images, GroundingDINO candidate grids, fused-cloud previews, perception views, grasp visualizations, and logs.

Generated NPZ files, PLY files, images, videos, logs, compiled extensions, and other runtime artifacts are excluded from Git.

## Third-Party Components

This repository contains project-local copies or subsets of third-party components used by the ThinkGrasp pipeline.

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

Current implemented / validated components include:

- MuJoCo Panda + Franka Hand scene
- 10 fixed five-object GSO clutter scenes
- RGB-D perception and calibrated world workspace
- high-resolution RAW workspace crop for GroundingDINO
- Qwen3-VL target selection and 3×3 preferred grasp location
- GroundingDINO localization with confidence + VLM-centroid soft ranking
- GraspNet inference
- target-region grasp filtering
- continuous angle + preferred-location grasp ranking
- four-view full-scene fallback grasping
- continuity-first Panda inverse kinematics
- JOINT_POSITION / q_ref grasp execution
- 1 cm Cartesian grasp / lift waypoints
- persistent force-stop during grasp descent
- stable-width gripper closing
- fixed joint-space transport
- simulator-side reward and task-success evaluation
- lift-failure release-before-home recovery
- retry-based closed-loop execution
- grasp / perception debugging outputs
- project-local native extension build workflow
- clean-clone standalone runtime design

The project is under active development as part of a bachelor thesis on VLM-based robotic grasping and simulation migration.

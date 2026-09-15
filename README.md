# ThinkGrasp-MuJoCo

A MuJoCo-based experimental platform for language-guided target grasping in cluttered scenes.

The project investigates how a vision-language model (VLM) can assist target localization, grasp selection, and closed-loop manipulation.

## Main Features

- VLM-based target selection and visual description
- VLM centroid prediction for spatial guidance
- GroundingDINO open-vocabulary candidate generation
- Centroid-guided target localization
- GraspNet 6-DoF grasp generation
- Geometry-only grasp selection
- VLM-only grasp selection
- VLM-guided grasp selection
- Closed-loop execution with scene re-perception
- Robot recovery and replanning
- Paired full-versus-baseline evaluation using the same MuJoCo scene state

## System Pipeline

```text
Language instruction + scene image
                |
                v
              VLM
                |
                +-- visual description
                +-- object centroid
                +-- preferred grasp region
                |
                v
         GroundingDINO localization
                |
                v
         GraspNet grasp generation
                |
                v
          Grasp candidate ranking
                |
                v
       MuJoCo robot execution
                |
                v
       Re-perception and replanning
```

## Main Components

### Core Runtime

- `run_closed_loop.py`  
  Main closed-loop execution script.

- `thinkgrasp_minimal_env.py`  
  MuJoCo and robosuite environment definition.

- `scene_bridge.py`  
  Scene and point-cloud data exchange.

- `vlm_bridge.py`  
  Interface between the VLM and the manipulation pipeline.

- `grasp_detector.py`  
  Grasp candidate processing and ranking.

- `graspnet_bridge.py`  
  GraspNet inference interface.

- `graspnet_config.py`  
  GraspNet-related configuration.

- `dual_view_recorder.py`  
  Execution video recording.

### Perception

- `run_groundingdino_inference.py`  
  GroundingDINO inference.

- `run_pointcloud_fusion.py`  
  Point-cloud fusion.

- `pybullet_pointcloud_fusion.py`  
  Point-cloud reconstruction utilities.

- `perception_viz.py`  
  Perception and grasp visualization.

- `run_graspnet_inference.py`  
  Standalone GraspNet inference.

### Experiments

- `run_end_to_end_comparison.py`  
  Paired full-versus-baseline evaluation.

- `run_closed_loop_batch.py`  
  Batch closed-loop evaluation.

- `run_grasp_selection_evaluation.py`  
  Geometry-only, VLM-only, and VLM-guided grasp-selection evaluation.

- `run_grasp_selection_batch.py`  
  Batch grasp-selection evaluation.

- `run_localization_comparison.py`  
  VLM-prompt DINO and VLM-guided DINO comparison.

- `run_localization_evaluation.py`  
  Localization evaluation.

- `run_vlm_language_check.py`  
  VLM language and clutter-awareness capability checks.

- `collect_scene11_localization_cases.py`  
  Scene generation for localization experiments.

## Environment

The system was developed and tested with:

- Python 3.10
- MuJoCo
- robosuite
- PyTorch
- GroundingDINO
- GraspNet
- Qwen3-VL

The exact package versions depend on the local Conda environment.

## Case Files

Case configurations are stored in the `cases/` directory.

Examples include:

```text
cases/case_scene01_white_ramekin.txt
cases/case_scene11_white_ramekin.txt
cases/case_scene11_gaming_mouse.txt
cases/case_scene11_mario_figure.txt
```

VLM capability-check cases and images are stored in:

```text
cases/vlm_language_cases/
```

## Running the Closed-Loop System

Example:

```bash
python run_closed_loop.py \
  --scene scene11 \
  --case cases/case_scene11_white_ramekin.txt
```

The default configuration runs the full VLM-guided pipeline.

## Running the Baseline

The baseline uses:

- the original target description;
- the highest-confidence GroundingDINO candidate;
- geometry-only grasp selection.

Example:

```bash
python run_closed_loop.py \
  --scene scene11 \
  --case cases/case_scene11_white_ramekin.txt \
  --evaluation-mode baseline
```

## Paired End-to-End Evaluation

The paired evaluation runs the full and baseline configurations on the same initial MuJoCo scene state.

Example:

```bash
python run_end_to_end_comparison.py \
  --scene scene11 \
  --case cases/case_scene11_white_ramekin.txt \
  --count 15 \
  --max-attempts 15 \
  --output-dir end_to_end_evaluation_paired15
```

For each trial:

```text
1. Generate one MuJoCo scene.
2. Save its initial state.
3. Run the full VLM-guided configuration.
4. Restore the saved state.
5. Run the baseline configuration.
```

The paired scene states are stored under:

```text
paired_scene_states/
```

## Grasp-Selection Evaluation

The grasp-selection evaluation generates grasp candidates and exports the selected grasp results without executing the complete task.

Example:

```bash
python run_grasp_selection_evaluation.py \
  --scene scene11 \
  --case cases/case_scene11_mario_figure.txt
```

The generated files may include:

- all grasp candidates;
- target-region point cloud;
- geometry-only selected grasp;
- VLM-guided selected grasp;
- VLM-only selected grasp;
- comparison metadata.

## Localization Evaluation

Example:

```bash
python run_localization_comparison.py \
  --scene scene11 \
  --case cases/case_scene11_white_ramekin.txt
```

The localization comparison evaluates:

- VLM-prompt DINO;
- VLM-guided DINO.

The guided configuration combines GroundingDINO confidence with the VLM-predicted centroid.

## VLM Capability Checks

The capability-check cases are located at:

```text
cases/vlm_language_cases/
```

They include:

- direct language-based target selection;
- indirect language understanding;
- clutter-aware object selection;
- obstruction removal reasoning.

## Scene Configuration

The project contains multiple fixed MuJoCo object sets.

Examples:

- `scene01`: less cluttered configuration;
- `scene11`: highly cluttered configuration.

The scene configuration is defined in:

```text
thinkgrasp_minimal_env.py
```

The target object and language instruction are defined in the corresponding case file.

## Output Management

Generated runtime outputs are intentionally excluded from Git.

Examples include:

```text
bridge_data/
closed_loop_outputs/
localization_evaluation/
grasp_selection_evaluation/
grasp_videos/
vlm_capability_outputs/
workspace_preview/
end_to_end_evaluation*/
end_to_end_test*/
```

Generated logs, videos, point clouds, temporary files, and MuJoCo state files are also excluded.

## Reproducibility Notes

For paired comparisons, the same initial MuJoCo state must be used by both configurations.

The paired evaluation therefore saves:

- MuJoCo joint positions;
- MuJoCo joint velocities;
- simulation state information.

The baseline restores this state before execution instead of using a newly generated random scene.

## Third-Party Components

This project uses third-party components including:

- robosuite;
- MuJoCo;
- GroundingDINO;
- GraspNet;
- PyTorch;
- scanned-object assets;
- external model checkpoints.

Please check the licenses of all third-party software, models, datasets, and scanned objects before redistribution.

## License

This repository contains research code. The licensing status of third-party components and assets remains subject to their original licenses.

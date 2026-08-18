from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent

graspnet_config = {
    "graspnet_checkpoint_path": str(
        PROJECT_DIR
        / "models"
        / "graspnet"
        / "logs"
        / "log_rs"
        / "checkpoint.tar"
    ),
    "refine_approach_dist": 0.01,
    "dist_thresh": 0.05,
    "angle_thresh": 15,
    "mask_thresh": 0.5,
}

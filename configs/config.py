import yaml
import json
from pathlib import Path


NDT_CONFIG_KEYS = {
    "voxel_size",
    "map_start_frame",
    "map_end_frame",
    "map_downsample_voxel",
    "min_points_per_voxel",
    "verbose",
    "translation_lr",
    "rotation_lr",
    "pcl_step_size",
    "pcl_transformation_epsilon",
    "pcl_outlier_ratio",
    "pcl_line_search_max_steps",
    "pcl_radius_chunk_size",
    "source_downsample_voxel",
    "pose_iters",
    "use_soft_nll",
    "noise_tr_xy",
    "noise_tr_z",
    "noise_roll_pitch",
    "noise_yaw",
    "covs_jitter",
    "n_neighbours",
    "device",
    "dtype",
    "kitti_root",
    "kitti_sequence",
}


def validate_config_keys(cfg: dict, allowed_keys: set[str]) -> None:
    if set(allowed_keys) != set(cfg):
        raise AttributeError("Wrong config keys")


def load_config_file(path: str | Path, *, allowed_keys: set[str] | None = None) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() in {".yml", ".yaml"}:
        with path.open("r") as f:
            data = yaml.safe_load(f) or {}
    elif path.suffix.lower() == ".json":
        with path.open("r") as f:
            data = json.load(f) or {}
    else:
        raise ValueError(f"Unsupported config extension: {path.suffix}")

    if not isinstance(data, dict):
        raise ValueError("Top-level config must be a mapping/dict.")
    if allowed_keys is not None:
        validate_config_keys(data, allowed_keys)
    return data

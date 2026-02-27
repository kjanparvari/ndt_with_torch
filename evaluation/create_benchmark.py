from __future__ import annotations

import datetime as dt
from pathlib import Path
import sys

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import load_config_file as load_config
from ndt.utils import sample_random_se3_transform
from kitti_dataset import KittiOdometryDataset


def create_benchmark(
    *,
    output: Path,
    config_path: Path | None,
    num_cases: int = 50,
    seed: int = 27,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> dict:
    config = load_config(config_path) if config_path else {}
    map_start_frame = int(config.get("map_start_frame", 60))
    map_end_frame = int(config.get("map_end_frame", 100))
    max_xy_translation = float(config.get("noise_tr_xy", 0.8))
    max_yaw_deg = float(config.get("noise_yaw", 10.0))
    max_z_translation = float(config.get("noise_tr_z", 0.2))
    max_roll_pitch_deg = float(config.get("noise_roll_pitch", 2.0))
    dataset_root = Path(config.get("kitti_root", "./kitti"))
    sequence = str(config.get("kitti_sequence", "01"))

    if not dataset_root.exists():
        raise FileNotFoundError(f"Kitti dataset not found at: {dataset_root}")
    dataset = KittiOdometryDataset(dataset_root, sequence=sequence)
    if map_start_frame > map_end_frame:
        raise ValueError(
            f"map_start_frame ({map_start_frame}) > map_end_frame ({map_end_frame})"
        )
    num_available = int(dataset.poses_in_lidar.shape[0])
    if num_available <= 0:
        raise RuntimeError("No frames available in dataset.")
    if map_start_frame < 0 or map_end_frame >= num_available:
        raise ValueError(
            f"Frame range [{map_start_frame}, {map_end_frame}] is outside valid range [0, {num_available - 1}]"
        )

    gen = torch.Generator(device=device)
    cases: list[dict] = []
    for case_id in range(num_cases):
        case_seed = seed + case_id
        gen.manual_seed(case_seed)
        frame = int(
            torch.randint(
                low=map_start_frame,
                high=map_end_frame + 1,
                generator=gen,
                size=(1,),
                device=device,
                dtype=dtype,
            ).item()
        )

        T_lidar_perturb = sample_random_se3_transform(
            max_xy_translation=max_xy_translation,
            max_yaw_deg=max_yaw_deg,
            max_z_translation=max_z_translation,
            max_roll_pitch_deg=max_roll_pitch_deg,
            gen=gen,
            seed=case_seed,
            dtype=dtype,
            device=device,
        )

        translation = T_lidar_perturb.translation().tolist()
        quat_xyzw = T_lidar_perturb.rotation().tolist()

        cases.append(
            {
                "id": int(case_id),
                "frame": int(frame),
                "translation_xyz": translation,
                "quaternion_xyzw": quat_xyzw,
            }
        )

    benchmark = {
        "meta": {
            "created_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat()
            + "Z",
            "config_path": "./" + str(config_path) if config_path else None,
            "noise_tr_xy": max_xy_translation,
            "noise_tr_z": max_z_translation,
            "noise_yaw": max_yaw_deg,
            "noise_roll_pitch": max_roll_pitch_deg,
        },
        "cases": cases,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        yaml.safe_dump(benchmark, f, sort_keys=False)
    return benchmark


def main() -> None:
    output_path = Path("./evaluation/benchmark.yaml")
    config_path = Path("./configs/default.yaml")
    benchmark = create_benchmark(
        output=output_path,
        config_path=config_path,
    )
    print(f"Benchmark written to {output_path} with {len(benchmark['cases'])} cases ")


if __name__ == "__main__":
    main()

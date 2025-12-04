from __future__ import annotations

import argparse
import math
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import sys
from typing import Optional
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import open3d as o3d
import pypose as pp
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kitti_dataset import KittiOdometryDataset
from configs.config import load_config_file
from metandt.utils import (
    inverse_composition_pose_errors,
    sample_random_se3_transform,
    inverse_composition_se3_log_mse,
)


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: int
    frame: int
    translation_xyz: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]


def _downsample_points(points_xyz: np.ndarray, voxel_size: float | None) -> np.ndarray:
    if voxel_size is None or voxel_size <= 0:
        return points_xyz
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz)
    downsampled = pcd.voxel_down_sample(voxel_size=voxel_size)
    return np.asarray(downsampled.points)


def _make_se3_from_xyz_quat(
    translation_xyz: tuple[float, float, float],
    quaternion_xyzw: tuple[float, float, float, float],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> pp.LieTensor:
    t = torch.tensor(translation_xyz, dtype=dtype, device=device)
    q = torch.tensor(quaternion_xyzw, dtype=dtype, device=device)
    q = q / torch.linalg.norm(q).clamp_min(1e-12)
    return pp.SE3(torch.cat([t, q], dim=-1))


def _load_benchmark(path: Path) -> tuple[dict[str, Any], list[BenchmarkCase]]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if "cases" not in data:
        raise ValueError(f"Benchmark file `{path}` has no `cases` entry.")

    cases: list[BenchmarkCase] = []
    for i, raw in enumerate(data["cases"]):
        case_id = int(raw.get("id", i))
        frame = int(raw["frame"])
        t = tuple(float(v) for v in raw["translation_xyz"])
        q = tuple(float(v) for v in raw["quaternion_xyzw"])
        if len(t) != 3:
            raise ValueError(f"Case {case_id}: `translation_xyz` must have length 3.")
        if len(q) != 4:
            raise ValueError(f"Case {case_id}: `quaternion_xyzw` must have length 4.")
        cases.append(
            BenchmarkCase(
                case_id=case_id,
                frame=frame,
                translation_xyz=t,
                quaternion_xyzw=q,
            )
        )
    return data.get("meta", {}), cases


def _load_config_from_benchmark(meta: dict[str, Any]) -> dict[str, Any]:
    raw_path = meta.get("config_path")
    if not raw_path:
        return {}
    config_path = Path(raw_path)
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()
    return load_config_file(config_path)


def evaluate_benchmark_cases(
    *,
    dataset: KittiOdometryDataset,
    cases: list[BenchmarkCase],
    align_fn: (
        Callable[[torch.Tensor, pp.LieTensor], tuple[pp.LieTensor, pp.LieTensor]] | None
    ) = None,
    align_fns: (
        dict[
            str,
            Callable[[torch.Tensor, pp.LieTensor], tuple[pp.LieTensor, pp.LieTensor]],
        ]
        | None
    ) = None,
    device: torch.device,
    dtype: torch.dtype,
    source_downsample_voxel: float | None,
    print_per_case: bool = True,
    before_case_fn: (
        Callable[[BenchmarkCase, torch.Tensor, pp.LieTensor], None] | None
    ) = None,
) -> dict[str, Any]:
    if not cases:
        raise ValueError("No benchmark cases found.")
    if align_fns is None:
        if align_fn is None:
            raise ValueError("Either align_fn or align_fns must be provided.")
        align_fns = {"model": align_fn}
        single_aligner = True
    else:
        if not align_fns:
            raise ValueError("align_fns must not be empty.")
        single_aligner = False

    rows: list[dict[str, Any]] = []
    totals = {name: {"loss": 0.0, "t": 0.0, "r": 0.0} for name in align_fns.keys()}

    for case in cases:
        source_points_in_lidar_np, T_world_from_lidar_gt_np = dataset.get_lidar_data(
            case.frame, "lidar", return_pose=True
        )
        source_points_in_lidar_np = _downsample_points(source_points_in_lidar_np, source_downsample_voxel)

        source_points_in_lidar = torch.tensor(
            source_points_in_lidar_np, dtype=dtype, device=device, requires_grad=False
        )
        T_world_from_lidar_gt = pp.from_matrix(
            torch.tensor(T_world_from_lidar_gt_np, dtype=dtype, device=device, requires_grad=False),
            ltype=pp.SE3_type,
        )

        T_lidar_perturb = _make_se3_from_xyz_quat(
            case.translation_xyz, case.quaternion_xyzw, device=device, dtype=dtype
        )
        T_world_from_lidar_init = T_world_from_lidar_gt * T_lidar_perturb

        if before_case_fn is not None:
            before_case_fn(case, source_points_in_lidar, T_world_from_lidar_init)

        for aligner_name, current_align_fn in align_fns.items():
            _, T_lidar_perturb_inv_pred = current_align_fn(source_points_in_lidar, T_world_from_lidar_init)

            loss = inverse_composition_se3_log_mse(T_lidar_perturb_inv_pred, T_lidar_perturb)
            t_err, r_err = inverse_composition_pose_errors(
                T_lidar_perturb_inv_pred, T_lidar_perturb, device=device
            )
            row = {
                "id": case.case_id,
                "frame": case.frame,
                "loss": float(loss.item()),
                "t_err_m": float(t_err),
                "r_err_deg": float(r_err),
            }
            if not single_aligner:
                row["aligner"] = aligner_name
            rows.append(row)

            totals[aligner_name]["loss"] += row["loss"]
            totals[aligner_name]["t"] += row["t_err_m"]
            totals[aligner_name]["r"] += row["r_err_deg"]

            if print_per_case:
                prefix = (
                    f"[case {case.case_id:03d}]"
                    if single_aligner
                    else f"[{aligner_name} case {case.case_id:03d}]"
                )
                print(
                    f"{prefix} frame={case.frame:04d} "
                    f"loss={row['loss']:.6f} t_err={row['t_err_m']:.6f}m "
                    f"r_err={row['r_err_deg']:.6f}deg"
                )

    def summarize_worst_cases(
        aligner_name: str,
    ) -> list[dict[str, float | int]]:
        aligner_rows = [
            row
            for row in rows
            if single_aligner or row.get("aligner") == aligner_name
        ]
        worst_count = max(1, math.ceil(0.10 * len(aligner_rows)))
        worst_rows = sorted(
            aligner_rows,
            key=lambda row: row["loss"],
            reverse=True,
        )[:worst_count]
        return [
            {
                "case_id": int(row["id"]),
                "err_tr_m": float(row["t_err_m"]),
                "err_rot_deg": float(row["r_err_deg"]),
                "loss": float(row["loss"]),
            }
            for row in worst_rows
        ]

    def print_worst_cases(
        aligner_name: str,
        worst_cases: list[dict[str, float | int]],
    ) -> None:
        prefix = "[eval]" if single_aligner else f"[eval:{aligner_name}]"
        print(f"{prefix} worst_10_percent_by_loss:")
        for row in worst_cases:
            print(
                f"{prefix}   case={row['case_id']:03d} "
                f"err_tr={row['err_tr_m']:.6f}m "
                f"err_rot={row['err_rot_deg']:.6f}deg "
                f"loss={row['loss']:.6f}"
            )

    per_aligner_cases = len(cases)
    summaries = {
        name: {
            "num_cases": per_aligner_cases,
            "mean_loss": total["loss"] / per_aligner_cases,
            "mean_t_err_m": total["t"] / per_aligner_cases,
            "mean_r_err_deg": total["r"] / per_aligner_cases,
            "worst_10_percent": summarize_worst_cases(name),
        }
        for name, total in totals.items()
    }
    if single_aligner:
        summary = next(iter(summaries.values()))
        print(
            "[eval] "
            f"cases={summary['num_cases']} "
            f"mean_loss={summary['mean_loss']:.6f} "
            f"mean_t_err={summary['mean_t_err_m']:.6f}m "
            f"mean_r_err={summary['mean_r_err_deg']:.6f}deg"
        )
        print_worst_cases("model", summary["worst_10_percent"])
    else:
        summary = summaries
        for name, current_summary in summaries.items():
            print(
                f"[eval:{name}] "
                f"cases={current_summary['num_cases']} "
                f"mean_loss={current_summary['mean_loss']:.6f} "
                f"mean_t_err={current_summary['mean_t_err_m']:.6f}m "
                f"mean_r_err={current_summary['mean_r_err_deg']:.6f}deg"
            )
            print_worst_cases(name, current_summary["worst_10_percent"])
    return {"summary": summary, "rows": rows}


def evaluate_metandt_random_cases(
    model: Any,
    tests_number: int = 20,
    writer: Optional[SummaryWriter] = None,
) -> float:
    total_loss = 0.0
    for i in range(tests_number):
        source_points_in_lidar, T_world_from_lidar_gt = model.sample_source_scan(seed=i, do_subsample=True)
        T_lidar_perturb = sample_random_se3_transform(
            max_xy_translation=model.noise_tr_xy,
            max_yaw_deg=model.noise_yaw,
            max_z_translation=model.noise_tr_z,
            max_roll_pitch_deg=model.noise_roll_pitch,
            gen=model.gen,
            seed=i,
            dtype=model.dtype,
            device=model.device,
        )
        source_points_in_lidar_perturbed = T_lidar_perturb.Act(source_points_in_lidar.detach().clone())
        source_points_in_world_init = T_world_from_lidar_gt.Act(source_points_in_lidar_perturbed.detach().clone())

        T_world_from_lidar_init = T_world_from_lidar_gt * T_lidar_perturb
        model.plot_gaussians(
            server=model._viser_server,
            show_target_points=True,
            source_points=source_points_in_world_init,
            clear_previous=True,
        )
        _, T_lidar_perturb_inv_pred, _ = model.align_pose_adam(source_points_in_lidar, T_world_from_lidar_init)
        t_err, r_err = inverse_composition_pose_errors(T_lidar_perturb_inv_pred, T_lidar_perturb)
        loss = inverse_composition_se3_log_mse(T_lidar_perturb_inv_pred, T_lidar_perturb)
        total_loss += float(loss.item())
        print(f"loss{i}: {float(loss.item())}, t_err: {t_err}, r_err: {r_err}")
        if writer:
            writer.add_scalar(f"eval_test/tr", t_err, i)
            writer.add_scalar(f"eval_test/rot", r_err, i)
            writer.add_scalar(f"eval_test/loss", loss, i)

    mean_loss = total_loss / tests_number
    print(f"[eval] total loss={mean_loss}")
    return mean_loss


def _metandt_alignment_functions(
    model: Any,
    aligner: str,
) -> dict[
    str, Callable[[torch.Tensor, pp.LieTensor], tuple[pp.LieTensor, pp.LieTensor]]
]:
    aligner = aligner.lower()
    if aligner not in {"adam", "lm", "pcl_torch", "both", "all"}:
        raise ValueError("aligner must be one of: adam, lm, pcl_torch, both, all")

    def align_pose_adam(
        source_points_in_lidar: torch.Tensor, T_world_from_lidar_init: pp.LieTensor
    ) -> tuple[pp.LieTensor, pp.LieTensor]:
        T_world_from_lidar_pred, T_lidar_perturb_inv_pred, _ = model.align_pose_adam(
            source_points_in_lidar, T_world_from_lidar_init
        )
        return T_world_from_lidar_pred, T_lidar_perturb_inv_pred

    def align_pose_lm(
        source_points_in_lidar: torch.Tensor, T_world_from_lidar_init: pp.LieTensor
    ) -> tuple[pp.LieTensor, pp.LieTensor]:
        T_world_from_lidar_pred, T_lidar_perturb_inv_pred, _ = model.align_pose_lm(
            source_points_in_lidar,
            T_world_from_lidar_init,
            vectorize=False,
        )
        return T_world_from_lidar_pred, T_lidar_perturb_inv_pred

    def align_pose_pcl_torch(
        source_points_in_lidar: torch.Tensor, T_world_from_lidar_init: pp.LieTensor
    ) -> tuple[pp.LieTensor, pp.LieTensor]:
        T_world_from_lidar_pred, T_lidar_perturb_inv_pred, _ = model.align_pose_pcl_torch(
            source_points_in_lidar,
            T_world_from_lidar_init,
        )
        return T_world_from_lidar_pred, T_lidar_perturb_inv_pred

    if aligner == "adam":
        return {"adam": align_pose_adam}
    if aligner == "lm":
        return {"lm": align_pose_lm}
    if aligner == "pcl_torch":
        return {"pcl_torch": align_pose_pcl_torch}
    if aligner == "all":
        return {
            "adam": align_pose_adam,
            "lm": align_pose_lm,
            "pcl_torch": align_pose_pcl_torch,
        }
    return {"adam": align_pose_adam, "lm": align_pose_lm}


def evaluate_metandt_benchmark(
    model: Any,
    benchmark_path: str = "evaluation/benchmark.yaml",
    writer: Optional[SummaryWriter] = None,
    aligner: str = "adam",
):
    dataset = model.dataset
    dtype = model.dtype
    device = model.device
    source_downsample_voxel = model.source_downsample_voxel
    if isinstance(benchmark_path, str):
        benchmark_path = Path(benchmark_path)
    print(
        f"[eval] evaluating MetaNDT on benchmark {benchmark_path} with aligner {aligner}..."
    )
    _, cases = _load_benchmark(benchmark_path)
    align_fns = _metandt_alignment_functions(model, aligner)

    def plot_initial_case(
        _case: BenchmarkCase,
        source_points_in_lidar: torch.Tensor,
        T_world_from_lidar_init: pp.LieTensor,
    ) -> None:
        source_points_in_world_init = T_world_from_lidar_init.Act(
            source_points_in_lidar.detach().clone()
        )
        model.plot_gaussians(
            server=model._viser_server,
            show_target_points=True,
            source_points=source_points_in_world_init,
            clear_previous=True,
        )

    if len(align_fns) == 1:
        results = evaluate_benchmark_cases(
            dataset=dataset,
            cases=cases,
            align_fn=next(iter(align_fns.values())),
            device=device,
            dtype=dtype,
            source_downsample_voxel=source_downsample_voxel,
            before_case_fn=plot_initial_case,
        )
    else:
        results = evaluate_benchmark_cases(
            dataset=dataset,
            cases=cases,
            align_fns=align_fns,
            device=device,
            dtype=dtype,
            source_downsample_voxel=source_downsample_voxel,
            before_case_fn=plot_initial_case,
        )

    if writer:
        summary = results["summary"]
        if len(align_fns) == 1:
            writer.add_scalar("eval_summary/mean_loss", summary["mean_loss"], 0)
            writer.add_scalar("eval_summary/mean_tr", summary["mean_t_err_m"], 0)
            writer.add_scalar("eval_summary/mean_rot", summary["mean_r_err_deg"], 0)
            for rank, row in enumerate(summary["worst_10_percent"]):
                writer.add_scalar("eval_worst_10/loss", row["loss"], rank)
                writer.add_scalar("eval_worst_10/tr", row["err_tr_m"], rank)
                writer.add_scalar("eval_worst_10/rot", row["err_rot_deg"], rank)
        else:
            for aligner_name, aligner_summary in summary.items():
                writer.add_scalar(
                    f"eval_summary/{aligner_name}/mean_loss",
                    aligner_summary["mean_loss"],
                    0,
                )
                writer.add_scalar(
                    f"eval_summary/{aligner_name}/mean_tr",
                    aligner_summary["mean_t_err_m"],
                    0,
                )
                writer.add_scalar(
                    f"eval_summary/{aligner_name}/mean_rot",
                    aligner_summary["mean_r_err_deg"],
                    0,
                )
                for rank, row in enumerate(aligner_summary["worst_10_percent"]):
                    writer.add_scalar(
                        f"eval_worst_10/{aligner_name}/loss", row["loss"], rank
                    )
                    writer.add_scalar(
                        f"eval_worst_10/{aligner_name}/tr", row["err_tr_m"], rank
                    )
                    writer.add_scalar(
                        f"eval_worst_10/{aligner_name}/rot",
                        row["err_rot_deg"],
                        rank,
                    )

        case_steps: dict[int, int] = {}
        for row in results["rows"]:
            if row["id"] not in case_steps:
                case_steps[row["id"]] = len(case_steps)
            step = case_steps[row["id"]]
            prefix = f"{row['aligner']}/" if "aligner" in row else ""
            writer.add_scalar(f"eval_test/{prefix}tr", row["t_err_m"], step)
            writer.add_scalar(f"eval_test/{prefix}rot", row["r_err_deg"], step)
            writer.add_scalar(f"eval_test/{prefix}loss", row["loss"], step)

    summary = results["summary"]
    if len(align_fns) == 1:
        return summary["mean_loss"]
    return summary


def _build_metandt(
    *,
    checkpoint: str | None,
    from_latest: bool,
) -> Any:
    from metandt import MetaNDT

    model = MetaNDT()
    if (checkpoint is not None) and not from_latest:
        model.load_checkpoint(checkpoint)
    elif (checkpoint is None) and from_latest:
        model.load_latest_checkpoint()
    elif (checkpoint is None) and not from_latest:
        model.build_gaussian_map()
    else:
        raise ValueError("from_latest and checkpoint cannot be set simultaneously!")
    return model


def _run_cli() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate registration models on a prepared benchmark."
    )
    parser.add_argument(
        "--benchmark", type=Path, default=Path("evaluation/benchmark.yaml")
    )
    parser.add_argument("--output-json", type=Path, default=None)

    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--from-latest", action="store_true")
    parser.add_argument(
        "--aligner",
        choices=["adam", "lm", "pcl_torch", "both", "all"],
        default="adam",
        help="MetaNDT aligner to evaluate.",
    )

    args = parser.parse_args()

    meta, cases = _load_benchmark(args.benchmark)
    config = _load_config_from_benchmark(meta)
    sequence = config.get("kitti_sequence", "01")
    dataset_root = Path(config.get("kitti_root", "./kitti"))
    dataset = KittiOdometryDataset(dataset_root, sequence=sequence)

    source_downsample_voxel = config.get("source_downsample_voxel", 0.2)

    model = _build_metandt(
        checkpoint=args.checkpoint,
        from_latest=args.from_latest,
    )
    selected_align_fns = _metandt_alignment_functions(model, args.aligner)
    if len(selected_align_fns) == 1:
        eval_align_fn = next(iter(selected_align_fns.values()))
        eval_align_fns = None
    else:
        eval_align_fn = None
        eval_align_fns = selected_align_fns

    results = evaluate_benchmark_cases(
        dataset=dataset,
        cases=cases,
        align_fn=eval_align_fn,
        align_fns=eval_align_fns,
        device=model.device,
        dtype=model.dtype,
        source_downsample_voxel=source_downsample_voxel,
    )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "model": "metandt",
                    "aligner": args.aligner,
                    "benchmark": str(args.benchmark),
                    "sequence": sequence,
                    "summary": results["summary"],
                    "rows": results["rows"],
                },
                f,
                indent=2,
            )
        print(f"[eval] wrote JSON report to {args.output_json}")


if __name__ == "__main__":
    _run_cli()

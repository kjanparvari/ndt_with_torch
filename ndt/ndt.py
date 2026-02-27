from __future__ import annotations

import datetime
from contextlib import nullcontext
from typing import Tuple, Optional

from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import math
import open3d as o3d
import pypose as pp
from torch.utils.tensorboard import SummaryWriter

from create_map import create_velodyne_map
from kitti_dataset import KittiOdometryDataset
from .kdtree import KDTree
from .utils import parse_dtype
from configs.config import NDT_CONFIG_KEYS, load_config_file
import viser


# Keys we allow to be set from config files
class SE3CorrectionModule(torch.nn.Module):
    """
    SE(3) pose as twist parameters:
      v: translation (3,)
      w: rotation (axis-angle: (3,)) (quaternion: (4,))
    forward() Return SE(3) element as a PyPose LieTensor.
    """

    def __init__(
        self,
        t_init: torch.Tensor | None = None,
        r_init: torch.Tensor | None = None,
        device: str | torch.device = None,
        # rotation_mode: str = "axis-angle",
        rotation_mode: str = "quaternion",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.rotation_mode = rotation_mode
        if self.rotation_mode not in ("quaternion", "axis-angle"):
            raise ValueError(
                "Rotation mode must be either 'axis-angle' or 'quaternion'"
            )

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)

        if t_init is None:
            t = torch.zeros(3, device=device, dtype=dtype)
        elif not isinstance(t_init, torch.Tensor):
            raise TypeError("t_init must be torch.Tensor or None")
        elif not t_init.shape == (3,):
            raise ValueError("t_init must be torch.Tensor in shape (3,)")
        else:
            with torch.no_grad():
                t = t_init.to(device=device, dtype=dtype).detach().clone()

        rotation_shape = (3,) if rotation_mode == "axis-angle" else (4,)
        if r_init is None:
            if rotation_mode == "quaternion":
                r = torch.zeros(4, device=device, dtype=dtype)
                r[3] = 1.0
            else:
                r = torch.zeros(3, device=device, dtype=dtype)  # For axis-angle
        elif not isinstance(r_init, torch.Tensor):
            raise TypeError("r_init must be torch.Tensor or None")
        elif not r_init.shape == rotation_shape:
            raise ValueError(
                f"r_init must be torch.Tensor in shape {rotation_shape} for rotation_mode={self.rotation_mode}"
            )
        else:
            with torch.no_grad():
                r = r_init.to(device=device, dtype=dtype).detach().clone()

        self.t = torch.nn.Parameter(t, requires_grad=True)
        self.r = torch.nn.Parameter(r, requires_grad=True)

    def forward(self) -> pp.LieTensor:
        """Return SE(3) element as a PyPose LieTensor."""
        if self.rotation_mode == "axis-angle":
            # pp.so3 receives a rotation vector and saves it as LieTensor type
            q = pp.so3(self.r).Exp().tensor()  # SO3 (quaternion, xyzw)
        else:  # "quaternion"
            r = self.r
            eps = 1e-8
            norm = torch.linalg.norm(r)
            norm = torch.clamp(norm, min=eps)
            q = r / norm
        T = pp.SE3(torch.cat([self.t, q], dim=-1))

        return T

    @torch.no_grad()
    def reset_pose(
        self,
        t_init: Optional[torch.Tensor] = None,
        r_init: Optional[torch.Tensor] = None,
    ) -> None:
        """Reset parameters; if None, resets that part to zero."""
        if t_init is None:
            self.t.zero_()
        else:
            if t_init.shape != self.t.shape:
                raise RuntimeError(f"Shape mismatch: {t_init.shape} != {self.t.shape}")
            self.t.copy_(t_init).to(device=self.t.device, dtype=self.t.dtype)
        if r_init is None:
            self.r.zero_()
            if self.rotation_mode == "quaternion":
                self.r[3] = 1.0
        else:
            if r_init.shape != self.r.shape:
                raise RuntimeError(f"Shape mismatch: {r_init.shape} != {self.r.shape}")
            self.r.copy_(r_init).to(device=self.r.device, dtype=self.r.dtype)


class NDTCorrectionResidualModule(torch.nn.Module):
    """LM residual model for estimating the inverse lidar perturbation pose."""

    def __init__(
        self,
        ndt_model: "NDTModel",
        source_points_in_lidar: torch.Tensor,
        T_world_from_lidar_init: pp.LieTensor,
        *,
        use_soft_nll: bool,
        n_neighbours: int,
        device: str | torch.device,
        dtype: torch.dtype,
    ):
        super().__init__()
        device = torch.device(device)
        self.ndt_model = ndt_model
        self.source_points_in_lidar = source_points_in_lidar.to(
            device=device, dtype=dtype
        )
        self.T_world_from_lidar_init = T_world_from_lidar_init.to(
            device=device, dtype=dtype
        )
        self.use_soft_nll = use_soft_nll
        self.n_neighbours = n_neighbours
        self.delta = pp.Parameter(pp.se3(torch.zeros(6, device=device, dtype=dtype)))

    def pose(self) -> pp.LieTensor:
        """Return the current SE(3) correction."""
        return self.delta.Exp()

    def forward(self) -> torch.Tensor:
        if not hasattr(self, "_calls"):
            self._calls = 0
        self._calls += 1
        if self.ndt_model.verbose:
            print(f"[Pose LM Residual] Call={self._calls}")

        T_lidar_perturb_inv_pred = self.pose()
        source_points_in_lidar_corrected = T_lidar_perturb_inv_pred.Act(
            self.source_points_in_lidar
        )
        source_points_in_world_pred = self.T_world_from_lidar_init.Act(
            source_points_in_lidar_corrected
        )
        residuals, _ = self.ndt_model.ndt_pose_residuals(
            source_points_in_world_pred,
            n_neighbours=self.n_neighbours,
            soft=self.use_soft_nll,
        )
        return residuals


class NDTModel:
    """Voxelized Gaussian model for a target point cloud."""

    # hyperparameters
    voxel_size: float
    map_start_frame: int
    map_end_frame: int
    map_downsample_voxel: int
    min_points_per_voxel: int
    verbose: bool
    kitti_root: str | Path
    kitti_sequence: str | Path

    scheduler_factor: float = 0.5
    scheduler_threshold: float = 5e-4
    scheduler_patience: int = 9

    translation_lr: float
    rotation_lr: float
    pcl_step_size: float
    pcl_transformation_epsilon: float
    pcl_outlier_ratio: float
    pcl_line_search_max_steps: int
    pcl_radius_chunk_size: int

    pose_iters: int

    use_soft_nll: bool
    source_downsample_voxel: int
    noise_tr_xy: float
    noise_tr_z: float
    noise_roll_pitch: float
    noise_yaw: float
    covs_jitter: float
    n_neighbours: int

    device: Optional[torch.device]
    dtype: Optional[torch.dtype]

    # Where to load hyperparameters from
    config_path: str | Path | None = "configs/default.yaml"
    dataset: KittiOdometryDataset
    gen: torch.Generator

    # Filled after build()
    target_points: torch.Tensor = None
    means: Optional[torch.Tensor] = None
    weights: Optional[torch.Tensor] = None
    covs: Optional[torch.Tensor] = None
    scales: Optional[torch.Tensor] = None
    rots: Optional[torch.Tensor] = None
    cov_invs: Optional[torch.Tensor] = None
    cov_sqrt_invs: Optional[torch.Tensor] = None
    log_dets: Optional[torch.Tensor] = None
    kdtree: Optional[KDTree] = None
    tb_writer: Optional[SummaryWriter] = None
    _viser_server: viser.ViserServer | None = None

    # ---------- construction ----------
    def __init__(self):
        self.apply_config()
        self.dataset = KittiOdometryDataset(
            self.kitti_root, sequence=self.kitti_sequence
        )
        print("Starting to create map pointcloud.")
        target_scan = create_velodyne_map(
            self.dataset,
            start_frame=self.map_start_frame,
            end_frame=self.map_end_frame,
            map_downsample_voxel=self.map_downsample_voxel,
        )
        print("Map pointcloud created.")
        if target_scan is None:
            raise RuntimeError("target_points must not be None")
        self.target_points = torch.tensor(
            target_scan, dtype=self.dtype, device=self.device
        )
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(27)
        self._viser_server = viser.ViserServer()

    def build_gaussian_map(self) -> None:
        """Voxelize and fit one Gaussian per voxel (mean, sample covariance) in pure Torch."""
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device, dtype = self.device, self.dtype

        print("Starting to build gaussian map.")
        pts = self.target_points
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("points must be (N, 3)")
        pts = pts.to(device=device, dtype=dtype)

        vs = float(self.voxel_size)
        if vs <= 0:
            raise ValueError("voxel_size must be > 0")

        # 1) Voxel indices (int64) on device
        voxel_idx = torch.floor(pts / vs).to(torch.int64)  # (N, 3)

        # 2) Group by voxel with unique+inverse+counts
        uniq_vox, inv, counts = torch.unique(
            voxel_idx, dim=0, return_inverse=True, return_counts=True
        )  # uniq_vox: (V,3), inv: (N,), counts: (V,)
        V = uniq_vox.shape[0]
        if V == 0:
            raise RuntimeError(
                "No voxels formed. Decrease voxel_size or check input points."
            )

        # 3) Per-voxel sums (for means)
        sums = torch.zeros((V, 3), device=device, dtype=dtype)
        sums.index_add_(0, inv, pts)  # sum of points per voxel
        n = counts.to(dtype=dtype).clamp_min(1)  # (V,)

        means = sums / n.unsqueeze(-1)  # (V, 3)

        # 4) Per-voxel second-moment sums for covariance
        #    sum_outer = Σ x x^T per voxel (vectorized as 9 components)
        outer = (pts.unsqueeze(2) * pts.unsqueeze(1)).reshape(-1, 9)  # (N, 9)
        sum_outer = torch.zeros((V, 9), device=device, dtype=dtype)
        sum_outer.index_add_(0, inv, outer)  # (V, 9)
        sum_outer = sum_outer.view(V, 3, 3)  # (V, 3, 3)

        # Sample covariance: Σ (x - μ)(x - μ)^T / (n-1)
        # Σ (x - μ)(x - μ)^T = Σ xx^T - n μ μ^T
        mu_outer = means.unsqueeze(2) @ means.unsqueeze(1)  # (V, 3, 3)
        numer = sum_outer - n.view(-1, 1, 1) * mu_outer
        denom = (n - 1).clamp_min(1)  # avoid div-by-zero for n=1
        covs = numer / denom.view(-1, 1, 1)  # (V, 3, 3)

        # 5) Filter by min_points_per_voxel
        valid = counts >= int(self.min_points_per_voxel)
        means = means[valid]
        covs = covs[valid]

        if means.numel() == 0:
            raise RuntimeError(
                "No voxels with enough points. Decrease voxel_size or min_points_per_voxel."
            )

        weights = torch.ones(means.shape[0], device=device, dtype=dtype)  # * (-1)

        self.means = means.detach()
        self.weights = weights.detach()

        # Eigen-decompose sample covariances to get initial rotation+scale.
        # Match PCL VoxelGridCovariance's near-singular covariance inflation.
        evals, evecs = torch.linalg.eigh(covs)  # (M,3), (M,3,3)
        min_covar_eigvalue = 0.01 * evals[:, -1:].clamp_min(1e-12)
        evals = torch.maximum(evals, min_covar_eigvalue)
        s0 = torch.sqrt(evals)  # stddevs

        dets = torch.det(evecs)  # [M]
        neg = dets < 0
        # flip ONE column for the negatives; here we flip the last column
        evecs[neg, :, 2] = -evecs[neg, :, 2]
        q0 = pp.mat2SO3(evecs).tensor()

        self.scales = torch.log(torch.expm1(s0).clamp_min(1e-12)).detach()
        self.rots = q0.detach()  # will be normalized on use

        # Build KD-tree on current means
        self.kdtree = KDTree(self.means.detach(), device=self.device, dtype=self.dtype)

        # Precompute inverses & log-dets
        self._refresh_gaussian_stats()

        if self.verbose:
            print(f"Built NDT with {self.num_gaussians()} Gaussian components")

    # --- every refresh ---
    def _refresh_gaussian_stats(self):
        s = F.softplus(self.scales)  # (M,3) positive
        # s = s.clamp(max=s_max)
        q = F.normalize(self.rots, dim=-1)
        R = pp.SO3(q).matrix()  # (M,3,3) orthonormal

        S = torch.diag_embed(s * s)  # diag(s^2)
        Sigma = R @ S @ R.transpose(-1, -2)  # Σ = R S R^T
        self.covs = Sigma

        Sinv = torch.diag_embed(1.0 / (s * s).clamp_min(self.covs_jitter))
        Ssqrt_inv = torch.diag_embed(torch.rsqrt((s * s).clamp_min(self.covs_jitter)))

        self.cov_invs = R @ Sinv @ R.transpose(-1, -2)  # Σ^{-1} = R S^{-1} R^T
        self.cov_sqrt_invs = R @ Ssqrt_inv @ R.transpose(-1, -2)

        self.log_dets = 2.0 * torch.log(s).sum(dim=-1)  # log|Σ| = 2∑log s  (det R = 1)

    # ---------- utilities ----------
    def num_gaussians(self) -> int:
        return 0 if self.means is None else int(self.means.shape[0])

    # ---------- visualization (optional) ----------
    def plot_gaussians(
        self,
        server: viser.ViserServer | None = None,
        sigma: float = 1.0,
        color: tuple[float, float, float] = (0.2, 0.6, 1.0),
        show_target_points: bool = False,
        source_points: torch.Tensor | None = None,
        color_by_weights: bool = True,
        weight_color_range: tuple[
            tuple[float, float, float], tuple[float, float, float]
        ] = (
            (0.2, 0.6, 1.0),
            (1.0, 0.2, 0.2),
        ),
        clear_previous: bool = False,
        source_only: bool = False,
    ):
        """Visualize current Gaussians as splats in Viser.

        Args:
            server: Optional existing ViserServer. If None, create/reuse self._viser_server.
            sigma: Scalar factor applied to covariance matrices (visual size of splats).
            color: Global color used when color_by_weights=False.
            show_target_points: Also render target point cloud as green points.
            source_points: Tensor of shape (N, 3) with points in world frame.
            color_by_weights: If True, color Gaussians according to self.weights.
            weight_color_range: (low_color, high_color) for weight-based colormap.
            clear_previous: If True, clear /gaussians, /target_points, /source_points first.
            source_only: If True, only clear & redraw /source_points; map stays untouched.
        """
        if self.means is None or self.covs is None:
            raise RuntimeError("Model not built yet (means/covs are None).")

        if source_points is not None and not isinstance(source_points, torch.Tensor):
            raise TypeError("source_points must be torch.Tensor")

        # ------------------------------------------------------------------
        # 1) Create / reuse a persistent Viser server
        # ------------------------------------------------------------------
        if server is None:
            if not hasattr(self, "_viser_server") or self._viser_server is None:
                self._viser_server = viser.ViserServer()
                print("Viser server started at http://localhost:8080")
            server = self._viser_server

        handles: dict[str, object] = {}

        if not hasattr(self, "_origin_frame"):
            self._origin_frame = server.scene.add_frame(
                name="/origin",
                wxyz=(1.0, 0.0, 0.0, 0.0),  # identity rotation (no rotation)
                position=(0.0, 0.0, 0.0),  # world origin
                # depending on your viser version:
                # either:
                show_axes=True,
                # or, in newer versions you can tweak sizes instead of show_axes:
                # axes_length=0.2,
                # axes_radius=0.005,
                # origin_radius=0.01,
            )
        # ------------------------------------------------------------------
        # 2) Source-only fast path for alignment visualization.
        # ------------------------------------------------------------------
        if source_only:
            # Only touch the /source_points node.
            if source_points is None:
                return handles

            try:
                server.scene.remove_by_name("/source_points")
            except Exception:
                pass

            src = source_points.detach().cpu().numpy().astype(np.float32)
            src_handle = server.scene.add_point_cloud(
                name="/source_points",
                points=src,
                colors=(1.0, 0.0, 0.0),  # red
                point_size=0.02,
            )
            handles["source_points"] = src_handle
            return handles

            # ------------------------------------------------------------------
            # 3) Full-map update (Gaussians + optional point clouds)
            # ------------------------------------------------------------------
        if clear_previous:
            for name in ["/gaussians", "/target_points", "/source_points"]:
                try:
                    server.scene.remove_by_name(name)
                except Exception:
                    pass

        means = self.means.detach().cpu().numpy().astype(np.float32)  # (M, 3)
        covs = self.covs.detach().cpu().numpy().astype(np.float32)  # (M, 3, 3)
        m = means.shape[0]

        # Colors for Gaussians
        if color_by_weights:
            if getattr(self, "weights", None) is None:
                raise RuntimeError("Weights not initialized but color_by_weights=True")
            w = self.weights.detach().cpu().numpy().reshape(-1)
            w_norm = (w - w.min()) / (w.max() - w.min() + 1e-12)
            c0 = np.array(weight_color_range[0], dtype=np.float32)
            c1 = np.array(weight_color_range[1], dtype=np.float32)
            rgbs = np.stack([(1.0 - wi) * c0 + wi * c1 for wi in w_norm], axis=0)
        else:
            base = np.array(color, dtype=np.float32)
            rgbs = np.tile(base[None, :], (m, 1))

        opacities = np.ones((m, 1), dtype=np.float32)
        covariances = (sigma**2) * covs

        splat_handle = server.scene.add_gaussian_splats(
            name="/gaussians",
            centers=means,
            covariances=covariances,
            rgbs=rgbs,
            opacities=opacities,
        )
        handles["gaussians"] = splat_handle

        # ------------------------------------------------------------------
        # 4) Optional target / source point clouds
        # ------------------------------------------------------------------
        if show_target_points and getattr(self, "target_points", None) is not None:
            tgt = self.target_points.detach().cpu().numpy().astype(np.float32)
            target_handle = server.scene.add_point_cloud(
                name="/target_points",
                points=tgt,
                colors=(0.0, 1.0, 0.0),  # green
                point_size=0.02,
            )
            handles["target_points"] = target_handle

        if source_points is not None:
            src = source_points.detach().cpu().numpy().astype(np.float32)
            src_handle = server.scene.add_point_cloud(
                name="/source_points",
                points=src,
                colors=(1.0, 0.0, 0.0),  # red
                point_size=0.02,
            )
            handles["source_points"] = src_handle

        return handles

    # ---------- likelihoods ----------

    def hard_ndt_nll(self, points: torch.Tensor) -> torch.Tensor:
        """Mean NLL of each point under its nearest Gaussian (non-differentiable assignment)."""
        if any(
            x is None for x in (self.means, self.cov_invs, self.log_dets, self.kdtree)
        ):
            raise RuntimeError("Model not built.")

        means = self.means.detach()
        invs = self.cov_invs.detach()
        dets = self.log_dets.detach()
        weights = self.weights.detach()

        # log π_k, with ∑_k exp(log_weights[k]) = 1
        # weights = torch.log_softmax(weights, dim=0)  # (K,)

        _, idxs = self.kdtree.query_nearest(points)  # (N,), (N,)
        mu = means[idxs].to(points.dtype)  # (N, D)
        invs = invs[idxs].to(points.dtype)  # (N, D, D)
        logdet = dets[idxs].to(points.dtype)  # (N,)
        sigmoid_w = weights[idxs].to(points.dtype).sigmoid()  # (N,)

        diff = points - mu  # (N, D)
        md2 = torch.einsum("ni,nij,nj->n", diff, invs, diff)  # (N,)
        D = means.shape[1]  # 3
        const = D * math.log(2.0 * math.pi)

        # print(torch.linalg.norm(invs, dim=(-2, -1)).max())

        # NLL(x) = -log[π_k* N(x | μ_k*, Σ_k*)] = 0.5*(...) - log π_k*
        nll = 0.5 * (1000 * (md2 * sigmoid_w) + 0.1 * logdet + const)
        # nll = 0.5 * (md2) - logw
        return nll.mean(), idxs.unique()

    def knn_ndt_nll(
        self,
        points: torch.Tensor,
        n_neighbours: int = 5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Soft NLL using a local k-NN mixture around each point."""
        if any(
            x is None for x in (self.means, self.cov_invs, self.log_dets, self.kdtree)
        ):
            raise RuntimeError("Model not built.")

        means = self.means.detach()
        cov_invs = self.cov_invs.detach()
        log_dets = self.log_dets.detach()
        weights = self.weights.detach()

        # weights = torch.log_softmax(weights, dim=0)  # (K,)

        _, idxs = self.kdtree.query(points, k=n_neighbours)  # (N, k) each

        mu = means[idxs].to(points.dtype)  # (N, k, D)
        inv = cov_invs[idxs].to(points.dtype)  # (N, k, D, D)
        logdet = log_dets[idxs].to(points.dtype)  # (N, k)
        sigmoid_w = weights[idxs].to(points.dtype).sigmoid()  # (N, k)

        diff = (
            points[:, None, :] - mu
        )  # (N, k, D) dists can be used here instead of computing diff manually
        md2 = torch.einsum("nkd,nkde,nke->nk", diff, inv, diff)  # (N, k)

        D = points.shape[-1]
        const = points.new_tensor(D * math.log(2.0 * math.pi))
        E = 0.5 * ((md2 * sigmoid_w) + logdet + const)  # (N, k)

        return E.mean(), idxs.unique()

    def ndt_pose_residuals(
        self,
        points: torch.Tensor,
        n_neighbours: int = 5,
        soft: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Whitened NDT residuals for PyPose least-squares optimizers.

        Returns residuals in shape `(N, k, 3)`. Their squared sum matches the
        pose-dependent term of the corresponding scalar NLL objective.
        """
        if any(
            x is None
            for x in (
                self.means,
                self.weights,
                self.cov_sqrt_invs,
                self.kdtree,
            )
        ):
            raise RuntimeError("Model not built.")

        means = self.means.detach()
        sqrt_invs = self.cov_sqrt_invs.detach()
        weights = self.weights.detach()

        query_points = points
        query_ctx = (
            torch.enable_grad() if not torch.is_grad_enabled() else nullcontext()
        )
        if not torch.is_grad_enabled() or not query_points.requires_grad:
            query_points = points.detach().clone().requires_grad_(True)

        if soft:
            with query_ctx:
                _, idxs = self.kdtree.query(query_points, k=n_neighbours)
        else:
            with query_ctx:
                _, idxs = self.kdtree.query_nearest(query_points)
            idxs = idxs.view(-1, 1)

        mu = means[idxs].to(points.dtype)
        sqrt_inv = sqrt_invs[idxs].to(points.dtype)
        sigmoid_w = weights[idxs].to(points.dtype).sigmoid()

        diff = points[:, None, :] - mu
        residuals = torch.einsum("nkde,nke->nkd", sqrt_inv, diff)
        residuals = residuals * torch.sqrt(sigmoid_w.clamp_min(1e-12)).unsqueeze(-1)

        if not soft:
            residuals = residuals * math.sqrt(1000.0)

        normalizer = math.sqrt(2.0 * points.shape[0] * idxs.shape[1])
        residuals = residuals / normalizer
        return residuals, idxs.unique()

    def _eval_run_name(self):
        t = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        return (
            f"{t}-v{self.voxel_size}-i{self.pose_iters}"
            f"-nt{self.noise_tr_xy}-nr{self.noise_yaw}"
            f"-lr{self.translation_lr}-{self.rotation_lr}"
        )

    def align_pose_adam(
        self,
        source_points_in_lidar: torch.Tensor,
        T_world_from_lidar_init: pp.LieTensor,
    ):
        """
        Args:
            source_points_in_lidar: Tensor of shape (N, 3), source points in lidar frame.
            T_world_from_lidar_init: Initial world-from-lidar pose,
                `(T_world_from_lidar_gt * T_lidar_perturb)`.
                `T_lidar_perturb` is in the lidar/source frame.
        Returns: Final Transformation

        T_lidar_perturb_inv_pred should estimate T_lidar_perturb.Inv()
        T_world_from_lidar_init * T_lidar_perturb_inv_pred should be T_world_from_lidar_gt

        """

        pose_module = SE3CorrectionModule(device=self.device, dtype=self.dtype)
        optimizer_pose = torch.optim.Adam(
            [
                {"params": [pose_module.t], "lr": self.translation_lr},
                {"params": [pose_module.r], "lr": self.rotation_lr},
            ]
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer_pose,
            mode="min",
            factor=self.scheduler_factor,
            patience=self.scheduler_patience,
            threshold=self.scheduler_threshold,
            threshold_mode="rel",
            min_lr=[self.translation_lr * 0.01, self.rotation_lr * 0.01],
        )
        for it in range(self.pose_iters):
            optimizer_pose.zero_grad()
            T_lidar_perturb_inv_pred = pose_module()
            source_points_in_lidar_corrected = T_lidar_perturb_inv_pred.Act(
                source_points_in_lidar
            )
            source_points_in_world_pred = T_world_from_lidar_init.Act(
                source_points_in_lidar_corrected
            )

            cur_lr_tr = optimizer_pose.param_groups[0]["lr"]
            cur_lr_rot = optimizer_pose.param_groups[1]["lr"]

            if self.use_soft_nll:
                loss, _ = self.knn_ndt_nll(
                    source_points_in_world_pred,
                    n_neighbours=self.n_neighbours,
                )
            else:
                loss, _ = self.hard_ndt_nll(source_points_in_world_pred)
            if self.verbose and it % 10 == 0:
                lr_info = f"lr_tr={cur_lr_tr:.6f}, lr_rot={cur_lr_rot:.6f}"
                print(f"[pose_loop {it}] loss={loss.item():.6f}\t{lr_info}")
                self.plot_gaussians(
                    server=self._viser_server,
                    show_target_points=True,
                    source_points=source_points_in_world_pred,
                    source_only=True,
                )
            loss.backward()
            optimizer_pose.step()
            # scheduler.threshold = self.scheduler_threshold * (
            #     cur_lr_tr / self.translation_lr
            # )
            scheduler.step(loss.detach().clone())

        with torch.no_grad():
            T_lidar_perturb_inv_pred = pose_module()
            source_points_in_world_pred = T_lidar_perturb_inv_pred.Act(
                source_points_in_lidar
            )
            source_points_in_world_pred = T_world_from_lidar_init.Act(
                source_points_in_world_pred
            )
            T_world_from_lidar_pred = T_world_from_lidar_init * T_lidar_perturb_inv_pred
        return (
            T_world_from_lidar_pred,
            T_lidar_perturb_inv_pred,
            source_points_in_world_pred,
        )

    def align_pose_lm(
        self,
        source_points_in_lidar: torch.Tensor,
        T_world_from_lidar_init: pp.LieTensor,
        *,
        max_steps: int | None = None,
        damping: float = 1e-3,
        vectorize: bool = True,
    ):
        """
        Align using PyPose Levenberg-Marquardt over whitened NDT residuals.

        This is an inference/evaluation path.
        """
        if any(
            x is None
            for x in (
                self.means,
                self.weights,
                self.cov_sqrt_invs,
                self.kdtree,
            )
        ):
            raise RuntimeError("Model not built.")

        # source_points_in_lidar = source_points_in_lidar[500:-500]
        # print(source_points_in_lidar.shape)
        # exit()
        steps = self.pose_iters if max_steps is None else int(max_steps)
        if steps < 1:
            raise ValueError("max_steps must be >= 1")

        pose_module = NDTCorrectionResidualModule(
            self,
            source_points_in_lidar,
            T_world_from_lidar_init,
            use_soft_nll=self.use_soft_nll,
            n_neighbours=self.n_neighbours,
            device=self.device,
            dtype=self.dtype,
        )
        optimizer_pose = pp.optim.LevenbergMarquardt(
            pose_module,
            strategy=pp.optim.strategy.Constant(damping),
            reject=0,
            vectorize=vectorize,
        )
        scheduler = pp.optim.scheduler.StopOnPlateau(
            optimizer_pose,
            steps=steps,
            patience=self.scheduler_patience,
            decreasing=self.scheduler_threshold,
            verbose=False,
        )

        while scheduler.continual():
            it = scheduler.steps
            if self.verbose:
                print(f"[LM align] Iteration {it+1}/{steps}")
            loss = optimizer_pose.step(())
            if self.verbose and it % 10 == 0:
                damping_info = optimizer_pose.param_groups[0].get("damping", damping)
                print(
                    f"[pose_lm_loop {it}] loss={loss.item():.6f}\t"
                    f"damping={damping_info:.6e}"
                )
                with torch.no_grad():
                    T_lidar_perturb_inv_pred = pose_module.pose()
                    source_points_in_lidar_corrected = T_lidar_perturb_inv_pred.Act(
                        source_points_in_lidar
                    )
                    source_points_in_world_pred = T_world_from_lidar_init.Act(
                        source_points_in_lidar_corrected
                    )
                    self.plot_gaussians(
                        server=self._viser_server,
                        show_target_points=True,
                        source_points=source_points_in_world_pred,
                        source_only=True,
                    )
            scheduler.step(loss)

        with torch.no_grad():
            T_lidar_perturb_inv_pred = pose_module.pose()
            source_points_in_world_pred = T_lidar_perturb_inv_pred.Act(
                source_points_in_lidar
            )
            source_points_in_world_pred = T_world_from_lidar_init.Act(
                source_points_in_world_pred
            )
            T_world_from_lidar_pred = T_world_from_lidar_init * T_lidar_perturb_inv_pred
        return (
            T_world_from_lidar_pred,
            T_lidar_perturb_inv_pred,
            source_points_in_world_pred,
        )

    def align_pose_pcl_torch(
        self,
        source_points_in_lidar: torch.Tensor,
        T_world_from_lidar_init: pp.LieTensor,
        *,
        max_steps: int | None = None,
    ):
        """
        Align using a PCL-style Torch NDT optimizer over the total pose.

        Unlike `align` and `align_pose_lm`, this path optimizes the full
        source-to-world transform initialized at `T_world_from_lidar_init`, then derives
        the source-frame correction from `T_world_from_lidar_init.Inv() * T_world_from_lidar_pred`.
        It is intended for evaluation and comparison with PCL NDT.
        """
        if any(x is None for x in (self.means, self.cov_invs)):
            raise RuntimeError("Model not built.")

        from .pcl_torch_ndt import PCLTorchNDTAligner

        steps = self.pose_iters if max_steps is None else int(max_steps)
        aligner = PCLTorchNDTAligner(
            means=self.means.detach(),
            cov_invs=self.cov_invs.detach(),
            weights=None if self.weights is None else self.weights.detach(),
            resolution=float(self.voxel_size),
            max_iterations=steps,
            step_size=float(self.pcl_step_size),
            transformation_epsilon=float(self.pcl_transformation_epsilon),
            outlier_ratio=float(self.pcl_outlier_ratio),
            line_search_max_steps=int(self.pcl_line_search_max_steps),
            radius_chunk_size=int(self.pcl_radius_chunk_size),
            verbose=bool(self.verbose),
        )
        return aligner.align(source_points_in_lidar, T_world_from_lidar_init)

    def evaluate(self, log_tb: bool = False):
        if any(
            x is None for x in (self.means, self.cov_invs, self.log_dets, self.kdtree)
        ):
            raise RuntimeError("Model not built.")
        from evaluation.eval import (
            evaluate_ndt_benchmark,
        )

        if not log_tb:
            writer = None
        elif not self.tb_writer:
            scheduler_info = f"f{self.scheduler_factor}-p{self.scheduler_patience}-t{self.scheduler_threshold}"
            tb_path = (
                Path("./runs")
                / "kitti"
                / f"{self._eval_run_name()}-{scheduler_info}--eval"
            )
            print(tb_path)
            writer = SummaryWriter(tb_path)
        else:
            writer = self.tb_writer
        return evaluate_ndt_benchmark(
            self,
            benchmark_path="./evaluation/benchmark.yaml",
            writer=writer,
            aligner="pcl_torch",
        )

    def apply_config(self) -> None:
        """Override hyperparameters from config file if config_path is set."""
        if self.config_path is None:
            return

        cfg = load_config_file(self.config_path, allowed_keys=NDT_CONFIG_KEYS)

        for k, v in cfg.items():
            setattr(self, k, v)

        if isinstance(self.device, str):
            self.device = torch.device(self.device)
        if isinstance(self.dtype, str):
            self.dtype = parse_dtype(self.dtype)
        if isinstance(self.kitti_root, str):
            self.kitti_root = Path(self.kitti_root)
        elif not isinstance(self.kitti_root, Path):
            raise TypeError("kitti_root must be str or pathlib.Path")
        if not self.kitti_root.exists():
            raise FileNotFoundError(f"Kitti dataset not found at: {self.kitti_root}")
        print(f"Config file {self.config_path} loaded successfully.")

    def load_saved_map(
        self,
        path: str | Path,
        *,
        map_location: Optional[str | torch.device] = None,
        strict: bool = True,
    ) -> dict:
        """
        Load saved Gaussian map tensors for inference.

        Args:
            path: Path to the saved map .pt file.
            map_location: Passed to torch.load (e.g., "cpu"). Defaults to self.device or CPU.
            strict: If True, enforce shape compatibility for means/covs/weights.

        Returns:
            The loaded saved-map dict.
        """
        # Decide where to load tensors, defaulting to CPU for portability.
        if map_location is None:
            map_location = "cpu"

        ckpt = torch.load(path, map_location=map_location)

        # Ensure device/dtype on the model
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = self.device
        dtype = self.dtype

        # --- Extract saved tensors (still on map_location) ---
        m_means = ckpt["model"]["means"]
        m_scales = ckpt["model"]["scales"]
        m_rots = ckpt["model"]["rots"]
        m_weights = ckpt["model"]["weights"]

        # Optional strict shape checks
        if strict and self.means is not None and self.means.shape != m_means.shape:
            raise ValueError(
                f"means shape mismatch: {self.means.shape} vs {m_means.shape}"
            )
        if (
            strict
            and self.weights is not None
            and self.weights.shape != m_weights.shape
        ):
            raise ValueError(
                f"weights shape mismatch: {self.weights.shape} vs {m_weights.shape}"
            )

        # --- Move to target device/dtype ---
        m_means = m_means.to(device=device, dtype=dtype)
        m_scales = m_scales.to(device=device, dtype=dtype)
        m_rots = m_rots.to(device=device, dtype=dtype)
        m_weights = m_weights.to(device=device, dtype=dtype)

        # --- Install tensors into the model as inference state ---
        names = ["means", "scales", "rots", "weights"]
        values = [m_means, m_scales, m_rots, m_weights]

        for name, v in zip(names, values):
            p = getattr(self, name)
            if p is None:
                setattr(self, name, v.detach().clone())
            else:
                with torch.no_grad():
                    p.copy_(v)

        # --- Rebuild KDTree & refresh derived stats ---
        self.kdtree = KDTree(self.means.detach(), device=self.device, dtype=self.dtype)
        self._refresh_gaussian_stats()  # recomputes cov_invs and log_dets from covs

        if self.verbose:
            print(
                f"[saved map] loaded <- {path} on device={device}, dtype={dtype}"
            )

        return ckpt

    def sample_source_scan(
        self, seed: int = None, do_subsample: bool = False
    ) -> Tuple[torch.Tensor, pp.LieTensor]:
        if seed is not None:
            self.gen.manual_seed(seed)
        # samples uniformly from [map_start_frame, map_end_frame]
        if self.map_start_frame > self.map_end_frame:
            raise ValueError(
                f"map_start_frame ({self.map_start_frame}) > map_end_frame ({self.map_end_frame})"
            )
        frame_number = torch.randint(
            low=self.map_start_frame,
            high=self.map_end_frame + 1,
            generator=self.gen,
            size=(1,),
            device=self.device,
        ).item()
        source_points_in_lidar, T_world_from_lidar_gt = self.dataset.get_lidar_data(
            frame_number, "lidar", return_pose=True
        )

        if do_subsample:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(source_points_in_lidar)
            downsampled_pcd = pcd.voxel_down_sample(
                voxel_size=self.source_downsample_voxel
            )  # .1: 58%  | .2: 34%
            # num_samples = max(1, source_points_in_lidar.shape[0] // self.source_downsample_factor)
            # downsampled_pcd = pcd.farthest_point_down_sample(num_samples=num_samples)
            source_points_in_lidar = np.asarray(downsampled_pcd.points)

        source_points_in_lidar = torch.tensor(
            source_points_in_lidar,
            dtype=self.dtype,
            device=self.device,
            requires_grad=False,
        )
        T_world_from_lidar_gt = torch.tensor(
            T_world_from_lidar_gt,
            dtype=self.dtype,
            device=self.device,
            requires_grad=False,
        )
        T_world_from_lidar_gt = pp.from_matrix(T_world_from_lidar_gt, ltype=pp.SE3_type)
        return source_points_in_lidar, T_world_from_lidar_gt

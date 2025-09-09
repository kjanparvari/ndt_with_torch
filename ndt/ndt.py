from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import torch

from .kdtree import KDTree
from .se3 import se3_exp, se3_transform_points


@dataclass
class NDTModel:
    """Voxelized Gaussian model for target point cloud."""

    voxel_size: float = 1.0
    min_points_per_voxel: int = 6
    covariance_damping: float = 1e-3
    device: Optional[torch.device] = None
    dtype: torch.dtype = torch.float32

    # Filled after build()
    means: Optional[torch.Tensor] = None
    cov_invs: Optional[torch.Tensor] = None
    log_dets: Optional[torch.Tensor] = None
    _kdtree: Optional[KDTree] = None

    def build(self, points: np.ndarray | torch.Tensor) -> None:
        if isinstance(points, torch.Tensor):
            pts = points.detach().cpu().numpy()
        else:
            pts = np.asarray(points, dtype=np.float64)
        assert pts.ndim == 2 and pts.shape[1] == 3, "points must be (N,3)"

        # Voxel indices
        vs = float(self.voxel_size)
        voxel_indices = np.floor(pts / vs).astype(np.int64)

        # Group points per voxel
        vox_to_pts: Dict[Tuple[int, int, int], list] = {}
        for idx, vidx in enumerate(map(tuple, voxel_indices)):
            vox_to_pts.setdefault(vidx, []).append(idx)

        means = []
        covs = []
        for vidx, idxs in vox_to_pts.items():
            if len(idxs) < self.min_points_per_voxel:
                continue
            P = pts[idxs]
            mu = P.mean(axis=0)
            X = P - mu
            # Sample covariance with damping
            cov = (X.T @ X) / max(1, len(P) - 1)
            cov = cov + (self.covariance_damping * np.eye(3))
            means.append(mu)
            covs.append(cov)

        if len(means) == 0:
            raise RuntimeError("No voxels with enough points. Decrease voxel_size or min_points_per_voxel.")

        means_np = np.asarray(means, dtype=np.float64)
        covs_np = np.asarray(covs, dtype=np.float64)

        invs = np.linalg.inv(covs_np)
        signs, logs = np.linalg.slogdet(covs_np)
        if not np.all(signs > 0):
            raise RuntimeError("Covariance not positive definite for some voxels.")

        device = self.device or torch.device("cpu")
        dtype = self.dtype

        self.means = torch.from_numpy(means_np).to(device=device, dtype=dtype)
        self.means.requires_grad_(True)
        self.cov_invs = torch.from_numpy(invs).to(device=device, dtype=dtype)
        self.log_dets = torch.from_numpy(logs).to(device=device, dtype=dtype)

        self._kdtree = KDTree(self.means)

    def num_components(self) -> int:
        return 0 if self.means is None else int(self.means.shape[0])

    @torch.no_grad()
    def assign_nearest(self, points: torch.Tensor) -> torch.Tensor:
        """Assign each point to nearest Gaussian by mean using KD-tree.
        Returns indices tensor (N,).
        """
        assert self._kdtree is not None, "Model not built"
        # pts_np = points.detach().cpu().numpy()
        dist, idxs = self._kdtree.query(points, k=1)
        return idxs

    def negative_log_likelihood(self, points: torch.Tensor) -> torch.Tensor:
        """Compute mean NLL of points under the nearest Gaussian component.

        Non-differentiable assignment is recomputed every call.
        """
        assert self.means is not None and self.cov_invs is not None and self.log_dets is not None
        idxs = self.assign_nearest(points)
        mu = self.means[idxs].to(points.dtype)
        inv = self.cov_invs[idxs].to(points.dtype)
        logdet = self.log_dets[idxs].to(points.dtype)

        diff = points - mu
        # Mahalanobis distance: sum over last dim
        md2 = (diff.unsqueeze(-2) @ inv @ diff.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        const = 3.0 * np.log(2.0 * np.pi)
        nll = 0.5 * (md2 + logdet + const)
        return nll.mean()

    def optimize_means(
            self,
            predicted_points: torch.Tensor,
            T_gt,
            T_est,
            opt: torch.optim.Optimizer,
            steps: int = 10,
            reassign_each_step: bool = True,
            verbose: bool = True,
    ) -> None:
        """
        Gradient-only update of Gaussian means, keeping covariances fixed.
        Rebuilds KD-tree each step (piecewise-constant assignment).
        """
        last = None
        for s in range(steps):
            opt.zero_grad(set_to_none=True)
            if reassign_each_step:
                self._kdtree = KDTree(self.means)
            # loss = self.negative_log_likelihood(predicted_points)  # NLL uses current nearest assignment
            loss = torch.nn.functional.l1_loss(T_est, T_gt)
            loss.backward()  # grads flow only into self.means
            opt.step()
            if verbose:
                cur = float(loss.item())
                print(f"[means] step {s:02d}: nll={cur:.6f}" + (f"  Δ={cur - last:.3e}" if last is not None else ""))
                last = cur
        # Keep KD-tree in sync with the new means
        self._kdtree = KDTree(self.means.detach().clone())


@dataclass
class NDTRegistration:
    model: NDTModel
    lr: float = 1e-1
    max_iters: int = 50
    tol: float = 1e-6
    verbose: bool = True

    def register(
            self,
            source_points: np.ndarray | torch.Tensor,
            init_xi: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = self.model.device or torch.device("cpu")
        dtype = self.model.dtype

        if isinstance(source_points, np.ndarray):
            src = torch.from_numpy(source_points).to(device=device, dtype=dtype)
        else:
            src = source_points.to(device=device, dtype=dtype)

        # Parameter: 6D twist xi
        # xi = torch.zeros(6, device=device, dtype=dtype) if init_xi is None else init_xi.to(device=device, dtype=dtype)
        v = torch.zeros(3, device=device, dtype=dtype, requires_grad=True)
        omega = torch.zeros(3, device=device, dtype=dtype, requires_grad=True)

        optimizer_pose = torch.optim.Adam([
            {'params': [v], 'lr': 1e-1},  # translation
            {'params': [omega], 'lr': 2e-2},  # rotation
        ])
        last_loss = None
        for it in range(self.max_iters):
            optimizer_pose.zero_grad(set_to_none=True)
            T = se3_exp(v, omega)
            src_tf = se3_transform_points(T, src)
            loss = self.model.negative_log_likelihood(src_tf)
            loss.backward()
            optimizer_pose.step()

            if self.verbose:
                print(f"iter {it:03d}: loss={loss.item():.6f}")

            if last_loss is not None and abs(last_loss - loss.item()) < self.tol:
                break
            last_loss = float(loss.item())

        with torch.no_grad():
            T_final = se3_exp(v.detach(), omega.detach())
        return T_final, torch.hstack([v.detach(), omega.detach()])

from __future__ import annotations

import math
from dataclasses import dataclass

import pypose as pp
import torch


@dataclass(frozen=True)
class PCLNDTScoreDerivatives:
    score: torch.Tensor
    gradient: torch.Tensor
    hessian: torch.Tensor | None


def pcl_pose_vec_to_matrix(pcl_pose_vec: torch.Tensor) -> torch.Tensor:
    """PCL convention: [tx, ty, tz, roll, pitch, yaw] -> T * Rx * Ry * Rz."""
    if pcl_pose_vec.shape != (6,):
        raise ValueError(
            f"Expected PCL pose vector shape (6,), got {tuple(pcl_pose_vec.shape)}"
        )

    tx, ty, tz, roll, pitch, yaw = pcl_pose_vec.unbind()
    one = torch.ones((), device=pcl_pose_vec.device, dtype=pcl_pose_vec.dtype)
    zero = torch.zeros((), device=pcl_pose_vec.device, dtype=pcl_pose_vec.dtype)

    cr, sr = torch.cos(roll), torch.sin(roll)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)

    rx = torch.stack(
        [
            torch.stack([one, zero, zero]),
            torch.stack([zero, cr, -sr]),
            torch.stack([zero, sr, cr]),
        ]
    )
    ry = torch.stack(
        [
            torch.stack([cp, zero, sp]),
            torch.stack([zero, one, zero]),
            torch.stack([-sp, zero, cp]),
        ]
    )
    rz = torch.stack(
        [
            torch.stack([cy, -sy, zero]),
            torch.stack([sy, cy, zero]),
            torch.stack([zero, zero, one]),
        ]
    )

    T_world_from_lidar_matrix = torch.eye(
        4, device=pcl_pose_vec.device, dtype=pcl_pose_vec.dtype
    )
    T_world_from_lidar_matrix[:3, :3] = rx @ ry @ rz
    T_world_from_lidar_matrix[:3, 3] = torch.stack([tx, ty, tz])
    return T_world_from_lidar_matrix


def pcl_matrix_to_pose_vec(T_world_from_lidar_matrix: torch.Tensor) -> torch.Tensor:
    """Inverse of `pcl_pose_vec_to_matrix` away from Euler singularities."""
    if T_world_from_lidar_matrix.shape != (4, 4):
        raise ValueError(
            "Expected transform matrix shape (4, 4), got "
            f"{tuple(T_world_from_lidar_matrix.shape)}"
        )

    R = T_world_from_lidar_matrix[:3, :3]
    pitch = torch.asin(torch.clamp(R[0, 2], -1.0, 1.0))
    cp = torch.cos(pitch)
    eps = torch.finfo(T_world_from_lidar_matrix.dtype).eps * 16.0

    if bool(torch.abs(cp) > eps):
        roll = torch.atan2(-R[1, 2], R[2, 2])
        yaw = torch.atan2(-R[0, 1], R[0, 0])
    else:
        roll = torch.zeros(
            (),
            device=T_world_from_lidar_matrix.device,
            dtype=T_world_from_lidar_matrix.dtype,
        )
        yaw = torch.atan2(R[1, 0], R[1, 1])

    return torch.cat(
        [T_world_from_lidar_matrix[:3, 3], torch.stack([roll, pitch, yaw])]
    )


class PCLTorchNDTAligner:
    """PCL-style Normal Distributions Transform optimizer in Torch.

    This mirrors PCL's inference process closely enough for experiments:
    transformed source points are scored against radius-neighbor target
    Gaussians, a Newton direction is solved from the 6D score Hessian, and a
    More-Thuente-style line search chooses the step length.
    """

    def __init__(
        self,
        *,
        means: torch.Tensor,
        cov_invs: torch.Tensor,
        weights: torch.Tensor | None = None,
        resolution: float,
        max_iterations: int,
        step_size: float = 1.5,
        transformation_epsilon: float = 1.0e-4,
        transformation_rotation_epsilon: float = 0.0,
        outlier_ratio: float = 0.55,
        line_search_max_steps: int = 10,
        radius_chunk_size: int = 1024,
        verbose: bool = False,
    ):
        if means.ndim != 2 or means.shape[1] != 3:
            raise ValueError(f"means must have shape (M, 3), got {tuple(means.shape)}")
        if cov_invs.shape != (means.shape[0], 3, 3):
            raise ValueError(
                "cov_invs must have shape (M, 3, 3) matching means; "
                f"got {tuple(cov_invs.shape)}"
            )
        if resolution <= 0:
            raise ValueError("resolution must be > 0")
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if step_size <= 0:
            raise ValueError("step_size must be > 0")
        if line_search_max_steps < 1:
            raise ValueError("line_search_max_steps must be >= 1")
        if radius_chunk_size < 1:
            raise ValueError("radius_chunk_size must be >= 1")
        if not (0.0 < outlier_ratio < 1.0):
            raise ValueError("outlier_ratio must be in (0, 1)")

        self.means = means.detach()
        self.cov_invs = cov_invs.detach()
        self.weights = None if weights is None else weights.detach()
        self.resolution = float(resolution)
        self.max_iterations = int(max_iterations)
        self.step_size = float(step_size)
        self.transformation_epsilon = float(transformation_epsilon)
        self.transformation_rotation_epsilon = float(transformation_rotation_epsilon)
        self.outlier_ratio = float(outlier_ratio)
        self.line_search_max_steps = int(line_search_max_steps)
        self.radius_chunk_size = int(radius_chunk_size)
        self.verbose = verbose

        c1 = 10.0 * (1.0 - self.outlier_ratio)
        c2 = self.outlier_ratio / (self.resolution**3)
        d3 = -math.log(c2)
        self.gauss_d1 = -math.log(c1 + c2) - d3
        self.gauss_d2 = -2.0 * math.log(
            (-math.log(c1 * math.exp(-0.5) + c2) - d3) / self.gauss_d1
        )

    def align(
        self,
        source_points_in_lidar: torch.Tensor,
        T_world_from_lidar_init: pp.LieTensor,
    ) -> tuple[pp.LieTensor, pp.LieTensor, torch.Tensor]:
        source_points_in_lidar, T_world_from_lidar_init, pcl_pose_vec = (
            self._prepare_alignment_inputs(
                source_points_in_lidar, T_world_from_lidar_init
            )
        )

        derivs = self._score_derivatives(
            source_points_in_lidar, pcl_pose_vec, compute_hessian=True
        )
        for iteration in range(self.max_iterations):
            pcl_pose_delta_vec = self._newton_delta(derivs.gradient, derivs.hessian)
            pcl_pose_delta_norm = torch.linalg.norm(pcl_pose_delta_vec)
            if (
                not torch.isfinite(pcl_pose_delta_norm)
                or float(pcl_pose_delta_norm.item()) == 0.0
            ):
                break

            step_dir = pcl_pose_delta_vec / pcl_pose_delta_norm
            step_len, step_dir, derivs = self._line_search(
                source_points_in_lidar=source_points_in_lidar,
                pcl_pose_vec=pcl_pose_vec,
                step_dir=step_dir,
                step_init=float(pcl_pose_delta_norm.item()),
                current_score=derivs.score,
                current_gradient=derivs.gradient,
            )
            pcl_pose_delta_vec = step_dir * step_len
            pcl_pose_vec = (pcl_pose_vec + pcl_pose_delta_vec).detach()

            if self.verbose:
                print(
                    f"[pcl_torch_ndt {iteration}] "
                    f"score={derivs.score.item():.6f} step={float(step_len):.6e}"
                )

            if self._has_converged(pcl_pose_delta_vec):
                break

        with torch.no_grad():
            return self._alignment_result(
                source_points_in_lidar, T_world_from_lidar_init, pcl_pose_vec
            )

    def _prepare_alignment_inputs(
        self,
        source_points_in_lidar: torch.Tensor,
        T_world_from_lidar_init: pp.LieTensor,
    ) -> tuple[torch.Tensor, pp.LieTensor, torch.Tensor]:
        source_points_in_lidar = source_points_in_lidar.to(
            device=self.means.device, dtype=self.means.dtype
        )
        T_world_from_lidar_init = T_world_from_lidar_init.to(
            device=self.means.device, dtype=self.means.dtype
        )
        T_world_from_lidar_init_matrix = T_world_from_lidar_init.matrix()
        pcl_pose_vec = pcl_matrix_to_pose_vec(
            T_world_from_lidar_init_matrix.detach()
        ).detach()
        return source_points_in_lidar, T_world_from_lidar_init, pcl_pose_vec

    def _alignment_result(
        self,
        source_points_in_lidar: torch.Tensor,
        T_world_from_lidar_init: pp.LieTensor,
        pcl_pose_vec: torch.Tensor,
    ) -> tuple[pp.LieTensor, pp.LieTensor, torch.Tensor]:
        T_world_from_lidar_pred_matrix = pcl_pose_vec_to_matrix(pcl_pose_vec)
        source_points_in_world_pred = self._transform_points(
            source_points_in_lidar, T_world_from_lidar_pred_matrix
        )
        T_world_from_lidar_pred = pp.from_matrix(
            T_world_from_lidar_pred_matrix, ltype=pp.SE3_type
        )
        T_lidar_perturb_inv_pred = (
            T_world_from_lidar_init.Inv() * T_world_from_lidar_pred
        )
        return (
            T_world_from_lidar_pred,
            T_lidar_perturb_inv_pred,
            source_points_in_world_pred,
        )

    def _score_derivatives(
        self,
        source_points_in_lidar: torch.Tensor,
        pcl_pose_vec: torch.Tensor,
        *,
        compute_hessian: bool,
    ) -> PCLNDTScoreDerivatives:
        pcl_pose_vec_var = pcl_pose_vec.detach().clone().requires_grad_(True)
        means = self.means.detach()
        cov_invs = self.cov_invs.detach()

        T_world_from_lidar_matrix = pcl_pose_vec_to_matrix(pcl_pose_vec_var)
        source_points_in_world = self._transform_points(
            source_points_in_lidar, T_world_from_lidar_matrix
        )

        with torch.no_grad():
            point_idxs, mean_idxs = self._radius_pairs(source_points_in_world.detach())

        score = self._score_from_pairs(
            source_points_in_world,
            point_idxs,
            mean_idxs,
            means,
            cov_invs,
            zero_like=pcl_pose_vec_var,
        )

        gradient = torch.autograd.grad(
            score,
            pcl_pose_vec_var,
            create_graph=compute_hessian,
            retain_graph=compute_hessian,
        )[0]

        hessian = None
        if compute_hessian:
            hessian = self._hessian_from_gradient(
                gradient,
                pcl_pose_vec_var,
                create_graph=False,
            )

        score = score.detach()
        gradient = gradient.detach()
        if hessian is not None:
            hessian = hessian.detach()

        return PCLNDTScoreDerivatives(
            score=score,
            gradient=gradient,
            hessian=hessian,
        )

    def _score_from_pairs(
        self,
        source_points_in_world: torch.Tensor,
        point_idxs: torch.Tensor,
        mean_idxs: torch.Tensor,
        means: torch.Tensor,
        cov_invs: torch.Tensor,
        *,
        zero_like: torch.Tensor,
    ) -> torch.Tensor:
        score = zero_like.sum() * 0.0
        if point_idxs.numel() == 0:
            return score

        diff = source_points_in_world[point_idxs] - means[mean_idxs]
        inv = cov_invs[mean_idxs]
        md2 = torch.einsum("ni,nij,nj->n", diff, inv, diff)
        exp_term = torch.exp(-0.5 * self.gauss_d2 * md2)
        valid = (
            torch.isfinite(exp_term)
            & (self.gauss_d2 * exp_term >= 0.0)
            & (self.gauss_d2 * exp_term <= 1.0)
        )
        return torch.where(valid, -self.gauss_d1 * exp_term, exp_term * 0.0).sum()

    @staticmethod
    def _hessian_from_gradient(
        gradient: torch.Tensor,
        pcl_pose_vec_var: torch.Tensor,
        *,
        create_graph: bool,
    ) -> torch.Tensor:
        if not gradient.requires_grad:
            return torch.zeros(
                6, 6, device=pcl_pose_vec_var.device, dtype=pcl_pose_vec_var.dtype
            )

        rows = []
        for i in range(6):
            if not gradient[i].requires_grad:
                row = torch.zeros_like(pcl_pose_vec_var)
            else:
                row = torch.autograd.grad(
                    gradient[i],
                    pcl_pose_vec_var,
                    retain_graph=True,
                    create_graph=create_graph,
                    allow_unused=True,
                )[0]
                if row is None:
                    row = torch.zeros_like(pcl_pose_vec_var)
            rows.append(row)
        hessian = torch.stack(rows, dim=0)
        return 0.5 * (hessian + hessian.transpose(0, 1))

    def _radius_pairs(
        self, source_points_in_world: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        radius_sqr = self.resolution * self.resolution
        point_idxs = []
        mean_idxs = []

        for start in range(0, source_points_in_world.shape[0], self.radius_chunk_size):
            stop = min(start + self.radius_chunk_size, source_points_in_world.shape[0])
            chunk = source_points_in_world[start:stop]
            diff = chunk[:, None, :] - self.means[None, :, :]
            dist_sqr = diff.square().sum(dim=-1)
            local_points, local_means = torch.nonzero(
                dist_sqr <= radius_sqr, as_tuple=True
            )
            if local_points.numel() == 0:
                continue
            point_idxs.append(local_points + start)
            mean_idxs.append(local_means)

        if not point_idxs:
            empty = torch.empty(
                0, device=source_points_in_world.device, dtype=torch.long
            )
            return empty, empty
        return torch.cat(point_idxs), torch.cat(mean_idxs)

    @staticmethod
    def _transform_points(
        source_points_in_lidar: torch.Tensor,
        T_world_from_lidar_matrix: torch.Tensor,
    ) -> torch.Tensor:
        return (
            source_points_in_lidar @ T_world_from_lidar_matrix[:3, :3].transpose(0, 1)
            + T_world_from_lidar_matrix[:3, 3]
        )

    def _newton_delta(
        self,
        gradient: torch.Tensor,
        hessian: torch.Tensor | None,
    ) -> torch.Tensor:
        if hessian is None:
            return torch.zeros_like(gradient)

        rhs = -gradient
        try:
            U, S, Vh = torch.linalg.svd(hessian)
            cutoff = torch.finfo(hessian.dtype).eps * max(hessian.shape) * S.max()
            Sinv = torch.where(S > cutoff, 1.0 / S, torch.zeros_like(S))
            return Vh.transpose(-2, -1) @ (Sinv * (U.transpose(-2, -1) @ rhs))
        except torch.linalg.LinAlgError:
            return torch.linalg.lstsq(hessian, rhs).solution

    def _line_search(
        self,
        *,
        source_points_in_lidar: torch.Tensor,
        pcl_pose_vec: torch.Tensor,
        step_dir: torch.Tensor,
        step_init: float,
        current_score: torch.Tensor,
        current_gradient: torch.Tensor,
    ) -> tuple[float, torch.Tensor, PCLNDTScoreDerivatives]:
        def score_fn(pose_vec: torch.Tensor, *, compute_hessian: bool):
            return self._score_derivatives(
                source_points_in_lidar,
                pose_vec,
                compute_hessian=compute_hessian,
            )

        step_len, step_dir, _, accepted = self._line_search_core(
            pcl_pose_vec=pcl_pose_vec,
            step_dir=step_dir,
            step_init=step_init,
            current_score=current_score,
            current_gradient=current_gradient,
            score_fn=score_fn,
            return_accepted=True,
        )
        if accepted is None:
            accepted = score_fn(pcl_pose_vec, compute_hessian=True)
        return step_len, step_dir, accepted

    def _line_search_core(
        self,
        *,
        pcl_pose_vec: torch.Tensor,
        step_dir: torch.Tensor,
        step_init: float,
        current_score: torch.Tensor,
        current_gradient: torch.Tensor,
        score_fn,
        return_accepted: bool,
    ) -> tuple[float, torch.Tensor, float, PCLNDTScoreDerivatives | None]:
        phi_0 = -float(current_score.item())
        d_phi_0 = -float(current_gradient.dot(step_dir).item())
        direction_sign = 1.0

        if d_phi_0 >= 0.0:
            if d_phi_0 == 0.0:
                accepted = (
                    score_fn(pcl_pose_vec, compute_hessian=True)
                    if return_accepted
                    else None
                )
                return 0.0, step_dir, direction_sign, accepted
            step_dir = -step_dir
            d_phi_0 = -d_phi_0
            direction_sign = -1.0

        mu = 1.0e-4
        nu = 0.9
        step_min = self.transformation_epsilon / 2.0
        step_max = self.step_size

        a_l = 0.0
        a_u = 0.0
        f_l = self._psi(a_l, phi_0, phi_0, d_phi_0, mu)
        g_l = self._dpsi(d_phi_0, d_phi_0, mu)
        f_u = f_l
        g_u = g_l

        a_t = min(max(step_init, step_min), step_max)
        open_interval = True
        interval_converged = False

        derivs = score_fn(pcl_pose_vec + step_dir * a_t, compute_hessian=False)
        phi_t = -float(derivs.score.item())
        d_phi_t = -float(derivs.gradient.dot(step_dir).item())
        psi_t = self._psi(a_t, phi_t, phi_0, d_phi_0, mu)
        d_psi_t = self._dpsi(d_phi_t, d_phi_0, mu)

        step_iterations = 0
        while (
            not interval_converged
            and step_iterations < self.line_search_max_steps
            and (psi_t > 0.0 or d_phi_t > -nu * d_phi_0)
        ):
            if open_interval:
                next_a = self._trial_value_selection(
                    a_l, f_l, g_l, a_u, f_u, g_u, a_t, psi_t, d_psi_t
                )
            else:
                next_a = self._trial_value_selection(
                    a_l, f_l, g_l, a_u, f_u, g_u, a_t, phi_t, d_phi_t
                )

            if not math.isfinite(next_a) or next_a == a_t:
                next_a = 0.5 * (a_t + a_l) if a_u == 0.0 else 0.5 * (a_l + a_u)

            a_t = min(max(next_a, step_min), step_max)
            derivs = score_fn(pcl_pose_vec + step_dir * a_t, compute_hessian=False)
            phi_t = -float(derivs.score.item())
            d_phi_t = -float(derivs.gradient.dot(step_dir).item())
            psi_t = self._psi(a_t, phi_t, phi_0, d_phi_0, mu)
            d_psi_t = self._dpsi(d_phi_t, d_phi_0, mu)

            if open_interval and (psi_t <= 0.0 and d_psi_t >= 0.0):
                open_interval = False
                f_l += phi_0 - mu * d_phi_0 * a_l
                g_l += mu * d_phi_0
                f_u += phi_0 - mu * d_phi_0 * a_u
                g_u += mu * d_phi_0

            if open_interval:
                interval_converged, a_l, f_l, g_l, a_u, f_u, g_u = (
                    self._update_interval(
                        a_l, f_l, g_l, a_u, f_u, g_u, a_t, psi_t, d_psi_t
                    )
                )
            else:
                interval_converged, a_l, f_l, g_l, a_u, f_u, g_u = (
                    self._update_interval(
                        a_l, f_l, g_l, a_u, f_u, g_u, a_t, phi_t, d_phi_t
                    )
                )
            step_iterations += 1

        accepted = (
            score_fn(pcl_pose_vec + step_dir * a_t, compute_hessian=True)
            if return_accepted
            else None
        )
        return float(a_t), step_dir, direction_sign, accepted

    @staticmethod
    def _psi(a: float, phi_a: float, phi_0: float, d_phi_0: float, mu: float) -> float:
        return phi_a - phi_0 - mu * d_phi_0 * a

    @staticmethod
    def _dpsi(d_phi_a: float, d_phi_0: float, mu: float) -> float:
        return d_phi_a - mu * d_phi_0

    @staticmethod
    def _update_interval(
        a_l: float,
        f_l: float,
        g_l: float,
        a_u: float,
        f_u: float,
        g_u: float,
        a_t: float,
        f_t: float,
        g_t: float,
    ) -> tuple[bool, float, float, float, float, float, float]:
        if f_t > f_l:
            return False, a_l, f_l, g_l, a_t, f_t, g_t
        if g_t * (a_l - a_t) > 0.0:
            return False, a_t, f_t, g_t, a_u, f_u, g_u
        if g_t * (a_l - a_t) < 0.0:
            return False, a_t, f_t, g_t, a_l, f_l, g_l
        return True, a_l, f_l, g_l, a_u, f_u, g_u

    @staticmethod
    def _trial_value_selection(
        a_l: float,
        f_l: float,
        g_l: float,
        a_u: float,
        f_u: float,
        g_u: float,
        a_t: float,
        f_t: float,
        g_t: float,
    ) -> float:
        if a_t == a_l and a_t == a_u:
            return a_t

        if a_t == a_l:
            condition = 4
        elif f_t > f_l:
            condition = 1
        elif g_t * g_l < 0.0:
            condition = 2
        elif abs(g_t) <= abs(g_l):
            condition = 3
        else:
            condition = 4

        if condition == 1:
            a_c = PCLTorchNDTAligner._cubic_minimizer(a_l, f_l, g_l, a_t, f_t, g_t)
            denom = g_l - (f_l - f_t) / (a_l - a_t)
            a_q = a_l - 0.5 * (a_l - a_t) * g_l / denom
            if abs(a_c - a_l) < abs(a_q - a_l):
                return a_c
            return 0.5 * (a_q + a_c)

        if condition == 2:
            a_c = PCLTorchNDTAligner._cubic_minimizer(a_l, f_l, g_l, a_t, f_t, g_t)
            a_s = a_l - (a_l - a_t) * g_l / (g_l - g_t)
            if abs(a_c - a_t) >= abs(a_s - a_t):
                return a_c
            return a_s

        if condition == 3:
            a_c = PCLTorchNDTAligner._cubic_minimizer(a_l, f_l, g_l, a_t, f_t, g_t)
            a_s = a_l - (a_l - a_t) * g_l / (g_l - g_t)
            a_next = a_c if abs(a_c - a_t) < abs(a_s - a_t) else a_s
            if a_t > a_l:
                return min(a_t + 0.66 * (a_u - a_t), a_next)
            return max(a_t + 0.66 * (a_u - a_t), a_next)

        return PCLTorchNDTAligner._cubic_minimizer(a_u, f_u, g_u, a_t, f_t, g_t)

    @staticmethod
    def _cubic_minimizer(
        a_l: float,
        f_l: float,
        g_l: float,
        a_t: float,
        f_t: float,
        g_t: float,
    ) -> float:
        denom = a_t - a_l
        if denom == 0.0:
            return a_t
        z = 3.0 * (f_t - f_l) / denom - g_t - g_l
        radicand = z * z - g_t * g_l
        if radicand < 0.0:
            radicand = 0.0
        w = math.sqrt(radicand)
        denom = g_t - g_l + 2.0 * w
        if denom == 0.0:
            return a_t
        return a_l + (a_t - a_l) * (w - g_l - z) / denom

    def _has_converged(self, pcl_pose_delta_vec: torch.Tensor) -> bool:
        translation_sqr = float(pcl_pose_delta_vec[:3].detach().square().sum().item())
        transform_eps = self.transformation_epsilon
        rotation_eps = self.transformation_rotation_epsilon
        cos_angle = -math.inf

        if rotation_eps > 0.0:
            delta_matrix = pcl_pose_vec_to_matrix(pcl_pose_delta_vec.detach())
            cos_angle = float(0.5 * (torch.trace(delta_matrix[:3, :3]).item() - 1.0))

        return (
            (
                transform_eps > 0.0
                and translation_sqr <= transform_eps
                and rotation_eps > 0.0
                and cos_angle >= rotation_eps
            )
            or (
                transform_eps <= 0.0
                and rotation_eps > 0.0
                and cos_angle >= rotation_eps
            )
            or (
                transform_eps > 0.0
                and translation_sqr <= transform_eps
                and rotation_eps <= 0.0
            )
        )

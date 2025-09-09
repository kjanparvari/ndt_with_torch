from __future__ import annotations

import torch


def _skew_symmetric_matrix(omega: torch.Tensor) -> torch.Tensor:
    wx, wy, wz = omega[..., 0], omega[..., 1], omega[..., 2]
    O = torch.zeros((*omega.shape[:-1], 3, 3), dtype=omega.dtype, device=omega.device)
    O[..., 0, 1] = -wz
    O[..., 0, 2] = wy
    O[..., 1, 0] = wz
    O[..., 1, 2] = -wx
    O[..., 2, 0] = -wy
    O[..., 2, 1] = wx
    return O


def se3_exp(v: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    """
    Exponential map from se(3) vector to SE(3) matrix.

    Args:
        xi: (..., 6) where xi[..., :3] = omega (rotation), xi[..., 3:] = v (translation tangent)

    Returns:
        T: (..., 4, 4) homogeneous transform
    """
    assert v.shape[-1] == 3 and omega.shape[-1] == 3, "xi must be (...,6)"
    # omega = xi[..., 0:3]
    # v = xi[..., 3:6]

    theta = torch.linalg.norm(omega, dim=-1, keepdim=True)
    small = (theta < 1e-8).squeeze(-1)

    K = _skew_symmetric_matrix(omega)

    # Rodrigues for SO(3)
    eye3 = torch.eye(3, dtype=v.dtype, device=v.device).expand(*omega.shape[:-1], 3, 3)

    # Avoid division by zero using series expansions
    A = torch.empty_like(theta)
    B = torch.empty_like(theta)
    C = torch.empty_like(theta)

    # Use Taylor series for small angles
    if small.any():
        th = theta[small]
        A_s = 1.0 - th.pow(2) / 6.0 + th.pow(4) / 120.0
        B_s = 0.5 - th.pow(2) / 24.0 + th.pow(4) / 720.0
        C_s = 1.0 / 6.0 - th.pow(2) / 120.0 + th.pow(4) / 5040.0
        A[small] = A_s
        B[small] = B_s
        C[small] = C_s
    if (~small).any():
        th = theta[~small]
        A_l = torch.sin(th) / th
        B_l = (1 - torch.cos(th)) / th.pow(2)
        C_l = (1 - A_l) / th.pow(2)
        A[~small] = A_l
        B[~small] = B_l
        C[~small] = C_l

    K2 = K @ K
    R = eye3 + A[..., None] * K + B[..., None] * K2

    # SO(3) left Jacobian for translation
    J = eye3 + B[..., None] * K + C[..., None] * K2
    t = (J @ v[..., None]).squeeze(-1)

    T = torch.eye(4, dtype=v.dtype, device=v.device).expand(*omega.shape[:-1], 4, 4).clone()
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    return T


def se3_transform_points(T: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Apply SE(3) transform to 3D points.

    Args:
        T: (..., 4, 4)
        points: (..., N, 3) or (N, 3) with broadcasting over batch dims of T
    Returns:
        transformed points with shape broadcasted to (..., N, 3)
    """
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    return (points @ R.transpose(-1, -2)) + t[..., None, :]

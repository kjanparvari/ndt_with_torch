import torch
import pypose as pp


def sample_random_se3_transform(
    max_xy_translation: float,
    max_z_translation: float,
    max_roll_pitch_deg: float,
    max_yaw_deg: float,
    gen: torch.Generator,
    seed: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> pp.lietensor:
    """
    Apply a random rigid transform (SE3) with the same logic as the original:
    - translation ~ U([-max_translation, max_translation]^3)
    - Euler rotations sampled independently in degrees in [-max_rotation_deg, max_rotation_deg]
    - Rotation order: R = Rz @ Ry @ Rx
    - Returns a PyPose SE3 LieTensor built from a 4x4 matrix.
    """
    gen.manual_seed(seed)

    def _rand_uniform(shape, low, high):
        r = torch.rand(shape, dtype=dtype, device=device, generator=gen)
        return low + (high - low) * r

    # ------------ Translation ------------
    # x, y: up to max_xy_translation
    # z: up to max_z_translation
    perturb_translation_xyz = torch.empty(3, dtype=dtype, device=device)
    perturb_translation_xyz[0:2] = _rand_uniform(
        (2,), -max_xy_translation, max_xy_translation
    )
    perturb_translation_xyz[2] = _rand_uniform(
        (), -max_z_translation, max_z_translation
    )

    # ------------ Rotation (roll/pitch small, yaw dominant) ------------

    roll_deg = _rand_uniform((), -max_roll_pitch_deg, max_roll_pitch_deg)  # Rx
    pitch_deg = _rand_uniform((), -max_roll_pitch_deg, max_roll_pitch_deg)  # Ry
    yaw_deg = _rand_uniform((), -max_yaw_deg, max_yaw_deg)  # Rz

    perturb_rpy_rad = torch.deg2rad(torch.stack((roll_deg, pitch_deg, yaw_deg)))

    cos_roll, cos_pitch, cos_yaw = torch.cos(perturb_rpy_rad)
    sin_roll, sin_pitch, sin_yaw = torch.sin(perturb_rpy_rad)

    # Rotation matrices around x, y, z
    R_from_roll = torch.eye(3, dtype=dtype, device=device)
    R_from_roll[1, 1] = cos_roll
    R_from_roll[1, 2] = -sin_roll
    R_from_roll[2, 1] = sin_roll
    R_from_roll[2, 2] = cos_roll

    R_from_pitch = torch.eye(3, dtype=dtype, device=device)
    R_from_pitch[0, 0] = cos_pitch
    R_from_pitch[0, 2] = sin_pitch
    R_from_pitch[2, 0] = -sin_pitch
    R_from_pitch[2, 2] = cos_pitch

    R_from_yaw = torch.eye(3, dtype=dtype, device=device)
    R_from_yaw[0, 0] = cos_yaw
    R_from_yaw[0, 1] = -sin_yaw
    R_from_yaw[1, 0] = sin_yaw
    R_from_yaw[1, 1] = cos_yaw

    # Compose full rotation matrix
    R_perturb = R_from_yaw @ R_from_pitch @ R_from_roll

    # ------------ Assemble 4x4 transform ------------
    T_perturb_matrix = torch.eye(4, dtype=dtype, device=device)
    T_perturb_matrix[:3, :3] = R_perturb
    T_perturb_matrix[:3, 3] = perturb_translation_xyz

    # Convert to PyPose SE3 LieTensor
    T_perturb = pp.from_matrix(T_perturb_matrix, ltype=pp.SE3_type)
    return T_perturb


def inverse_composition_se3_log_mse(
    T_perturb_inv_pred: pp.lietensor, T_perturb_gt: pp.lietensor
) -> torch.Tensor:
    w_rot = 1.0
    w_trans = 1.0
    perturb_error_se3_vec = (
        T_perturb_inv_pred * T_perturb_gt
    ).Log().tensor()  # right-invariant / se3: [rho(3), phi(3)]
    translation_error_vec = perturb_error_se3_vec[:3]
    rotation_error_vec = perturb_error_se3_vec[3:]
    inverse_composition_loss = (
        w_trans * (translation_error_vec**2).sum(-1)
        + w_rot * (rotation_error_vec**2).sum(-1)
    )
    # loss = w_trans * translation_error_vec.sum(-1) + w_rot * rotation_error_vec.sum(-1)
    return inverse_composition_loss


def inverse_composition_pose_errors(
    T_perturb_inv_pred, T_perturb_gt=None, device="cpu"
):
    if T_perturb_gt is None:
        T_perturb_gt = pp.identity_SE3(device=device)
    T_error = T_perturb_inv_pred * T_perturb_gt
    T_error_matrix = T_error.matrix()  # (..., 4, 4) or (4, 4)
    # Handle both batched and non-batched cases
    if T_error_matrix.ndim == 3:
        # Batched case: (batch, 4, 4)
        translation_error_vec = T_error_matrix[..., :3, 3]  # (batch, 3)
        R_error = T_error_matrix[..., :3, :3]  # (batch, 3, 3)
        t_err_m = torch.linalg.norm(translation_error_vec, dim=-1)  # (batch,)
        rotation_trace = (
            R_error[..., 0, 0] + R_error[..., 1, 1] + R_error[..., 2, 2]
        )  # (batch,)
    else:
        # Non-batched case: (4, 4)
        translation_error_vec = T_error_matrix[:3, 3]  # (3,)
        R_error = T_error_matrix[:3, :3]  # (3, 3)
        t_err_m = torch.linalg.norm(translation_error_vec, dim=-1)  # scalar or (1,)
        rotation_trace = R_error[0, 0] + R_error[1, 1] + R_error[2, 2]  # scalar
    rotation_cos_angle = torch.clamp((rotation_trace - 1.0) / 2.0, -1.0, 1.0)
    r_err_deg = torch.arccos(rotation_cos_angle).abs() * (180.0 / torch.pi)
    return t_err_m.item(), r_err_deg.item()


def parse_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "half": torch.float16,
        "float": torch.float32,
        "double": torch.float64,
    }
    try:
        return mapping[dtype_name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {dtype_name}") from exc

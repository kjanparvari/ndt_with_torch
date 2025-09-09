from __future__ import annotations

import torch
from kitti_dataset import KittiOdometryDataset
import os
import open3d as o3d
from create_map import subsample_map, apply_random_transform

from ndt import NDTModel, NDTRegistration, se3_exp, se3_transform_points


def main():
    os.putenv('XDG_SESSION_TYPE', 'x11')
    data_path = './kitti/'  # Adjust this to your KITTI dataset path
    if not os.path.exists(data_path):
        raise FileNotFoundError('Kitti dataset not found.')
    dataset = KittiOdometryDataset(data_path, sequence='01')
    device = torch.device("cuda")

    # Create map for sequence 01
    sequence = '01'
    start_frame = 50  # Starting frame
    end_frame = 100  # Ending frame

    source_frame = 75  # Index for source scan

    target_scan = dataset.get_lidar_data(source_frame, coordinate_system='world')
    source_scan, R_gt, tr_gt = apply_random_transform(target_scan.copy(), max_translation=3., max_rotation_deg=8.0,
                                                      seed=42)
    T_gt = torch.eye(4, dtype=torch.float32, device=device).clone()
    T_gt[:3, :3] = torch.tensor(R_gt, dtype=torch.float32, device=device)
    T_gt[:3, 3] = torch.tensor(tr_gt, dtype=torch.float32, device=device)

    source_scan = subsample_map(source_scan, voxel_size=.1)

    source_scan_torch = torch.tensor(source_scan, device="cuda", dtype=torch.float32)
    # target_scan_torch = torch.tensor(target_scan, device="cuda", dtype=torch.float32)

    model = NDTModel(voxel_size=5., min_points_per_voxel=6, covariance_damping=1e-3, device=device,
                     dtype=torch.float32)
    model.build(target_scan)
    print(f"Built NDT with {model.num_components()} Gaussian components")

    reg = NDTRegistration(model=model, lr=5e-2, max_iters=200, tol=1e-7, verbose=True)
    # T_est, xi_est = reg.register(source_scan_torch)
    T_est, xi_est = reg.register( source_scan_torch)

    T_expected = torch.linalg.inv(T_gt)
    # print("\nGround truth xi (applied to src):", xi_gt)
    # print("Estimated xi (to align to target):", xi_est)
    print("Expected T (inverse of GT):\n", T_expected)
    print("Estimated T:\n", T_est)

    # Report pose error relative to inverse of GT
    with torch.no_grad():
        dT = torch.linalg.inv(T_expected) @ T_est  # should be close to identity
        R_err = dT[:3, :3]
        t_err = dT[:3, 3]
        cos_angle = torch.clamp((torch.trace(R_err) - 1) / 2.0, -1.0, 1.0)
        rot_err = torch.arccos(cos_angle)
        trans_err = torch.linalg.norm(t_err)
        print(f"Rotation error (rad): {rot_err.item():.6f}, translation error: {trans_err.item():.6f}")

    source_pc = o3d.geometry.PointCloud()
    source_pc.points = o3d.utility.Vector3dVector(source_scan)
    # source_pc.points = o3d.utility.Vector3dVector(source_scan)
    target_pc = o3d.geometry.PointCloud()
    target_pc.points = o3d.utility.Vector3dVector(target_scan)

    # # Transform and visualize
    source_pc.transform(T_est.cpu().numpy())
    o3d.visualization.draw_geometries([source_pc.paint_uniform_color([1, 0, 0]),
                                       target_pc.paint_uniform_color([0, 1, 0])])


if __name__ == "__main__":
    main()

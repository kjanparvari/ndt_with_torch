import numpy as np
import torch
from matplotlib import pyplot as plt
from kitti_dataset import KittiOdometryDataset
import open3d as o3d
import os


def create_velodyne_map(dataset, start_frame, end_frame, map_downsample_voxel=10):
    """
    Create a Velodyne map for the given sequence.
    
    Args:
    dataset (KittiOdometryDataset): Dataset object
    start_frame (int): Starting frame index
    end_frame (int): Ending frame index (inclusive)
    downsample_factor (int): Factor to downsample points (to reduce memory usage)
    
    Returns:
    np.array: Accumulated point cloud map
    """
    accumulated_points = []

    for frame in range(start_frame, end_frame + 1):
        # Get LIDAR data and pose
        lidar_data = dataset.get_lidar_data(frame, coordinate_system='world')
        # Downsample points
        # down_np = lidar_data[::downsample_factor]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(lidar_data)
        # downsampled_pcd = pcd.farthest_point_down_sample(num_samples=lidar_data.shape[0] // downsample_factor)
        downsampled_pcd = pcd.voxel_down_sample(voxel_size=map_downsample_voxel)
        down_np = np.asarray(downsampled_pcd.points)

        accumulated_points.append(down_np)

    _map = np.vstack(accumulated_points)
    return _map


def subsample_map(points, voxel_size=0.5):
    """
    Subsample the map using voxel grid downsampling.
    
    Args:
    points (torch.Tensor): Nx3 tensor of points
    voxel_size (float): Size of voxel grid for downsampling
    
    Returns:
    torch.Tensor: Downsampled points
    """
    if not isinstance(points, torch.Tensor):
        raise TypeError('points must be a torch.Tensor')
    points_np = points.detach().cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)
    # downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    downsampled_pcd = pcd.farthest_point_down_sample(num_samples=points_np.shape[0]//3)
    down_np = np.asarray(downsampled_pcd.points)
    return torch.from_numpy(down_np).to(dtype=points.dtype, device=points.device)


def crop_middle_box(points: torch.Tensor, keep_ratio: float = 0.5) -> torch.Tensor:
    if not isinstance(points, torch.Tensor):
        raise TypeError('points must be a torch.Tensor')
    # keep_ratio in (0,1]; 0.5 keeps the central 50% along each axis
    mins, _ = points.min(dim=0)
    maxs, _ = points.max(dim=0)
    span = maxs - mins
    mid = (maxs + mins) * 0.5
    half = span * (keep_ratio * 0.5)
    lo, hi = mid - half, mid + half
    m = ((points >= lo) & (points <= hi)).all(dim=-1)
    return points[m]


def plot_map_2d(points, source_points=None):
    """
    Plot the 3D point cloud map in 2D.
    
    Args:
    points (np.array): Nx3 array of map points
    source_points (np.array, optional): Mx3 array of source scan points
    """
    plt.figure(figsize=(10, 10))

    # Plot map points in gray
    plt.scatter(points[:, 0], points[:, 1], s=0.1, color='gray', label='Map')

    # Plot source scan if provided
    if source_points is not None:
        plt.scatter(source_points[:, 0], source_points[:, 1], s=0.5, color='red', label='Source Scan')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Velodyne Map and Source Scan')
    plt.legend()
    plt.axis('equal')
    plt.show()


def plot_map_o3d(points, source_points=None):
    """
    Plot the 3D point cloud map using Open3D.
    
    Args:
    points (np.array): Nx3 array of map points
    source_points (np.array, optional): Mx3 array of source scan points
    """
    # Create Open3D point cloud object for map
    map_pcd = o3d.geometry.PointCloud()
    map_pcd.points = o3d.utility.Vector3dVector(points)

    # Set map color to gray
    map_pcd.paint_uniform_color([0.7, 0.7, 0.7])  # Gray color

    geometries = [map_pcd]

    # Add source scan if provided
    if source_points is not None:
        source_pcd = o3d.geometry.PointCloud()
        source_pcd.points = o3d.utility.Vector3dVector(source_points)
        source_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # Red color
        geometries.append(source_pcd)

    # Visualize point cloud
    o3d.visualization.draw_geometries(geometries)


if __name__ == "__main__":
    os.putenv('XDG_SESSION_TYPE', 'x11')
    data_path = './kitti/'  # Adjust this to your KITTI dataset path
    dataset = KittiOdometryDataset(data_path, sequence='01')

    # Create map for sequence 01
    sequence = '01'
    start_frame = 50  # Starting frame
    end_frame = 100  # Ending frame

    map_points = create_velodyne_map(dataset, start_frame, end_frame)

    # Subsample the map
    subsampled_map = subsample_map(map_points, voxel_size=.5)

    # Get a source scan to visualize
    source_frame = 75  # Index for source scan
    source_scan = dataset.get_lidar_data(source_frame, coordinate_system='world')

    print(f"Total points in original map: {map_points.shape[0]}")
    print(f"Total points in subsampled map: {subsampled_map.shape[0]}")
    print(f"Total points in source scan: {source_scan.shape[0]}")

    # Plot the map and source scan using Open3D
    plot_map_o3d(subsampled_map, source_scan)

import numpy as np
from matplotlib import pyplot as plt
from kitti_dataset import KittiOdometryDataset
import open3d as o3d
import os

def create_velodyne_map(dataset, start_frame, end_frame, downsample_factor=10):
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
        lidar_data = lidar_data[::downsample_factor]

        accumulated_points.append(lidar_data)

    _map = np.vstack(accumulated_points)
    return _map


def subsample_map(points, voxel_size=0.5):
    """
    Subsample the map using voxel grid downsampling.
    
    Args:
    points (np.array): Nx3 array of points
    voxel_size (float): Size of voxel grid for downsampling
    
    Returns:
    np.array: Downsampled points
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return np.asarray(downsampled_pcd.points)


def apply_random_transform(points, max_translation=2.0, max_rotation_deg=5.0, seed=0):
    """
    Apply a random rigid transform to a point cloud.

    Args:
    - points (np.ndarray): Nx3 point cloud
    - max_translation (float): Max translation in meters
    - max_rotation_deg (float): Max rotation in degrees

    Returns:
    - transformed_points (np.ndarray): Transformed Nx3 point cloud
    """
    # Create a seeded Generator instance
    rnd_gen = np.random.default_rng(seed)

    # Random translation vector in range [-max_translation, max_translation]
    translation = rnd_gen.uniform(-max_translation, max_translation, size=(3,))

    # Random small rotation in degrees
    angles = np.radians(rnd_gen.uniform(-max_rotation_deg, max_rotation_deg, size=(3,)))

    # Rotation matrices around x, y, z axes
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(angles[0]), -np.sin(angles[0])],
        [0, np.sin(angles[0]), np.cos(angles[0])]
    ])
    Ry = np.array([
        [np.cos(angles[1]), 0, np.sin(angles[1])],
        [0, 1, 0],
        [-np.sin(angles[1]), 0, np.cos(angles[1])]
    ])
    Rz = np.array([
        [np.cos(angles[2]), -np.sin(angles[2]), 0],
        [np.sin(angles[2]), np.cos(angles[2]), 0],
        [0, 0, 1]
    ])

    # Compose full rotation matrix
    R = Rz @ Ry @ Rx
    # R = Rz

    # translation[2] = 0.
    # Apply transform
    transformed_points = (R @ points.T).T + translation

    print(f"Initial translation: {translation}")
    print(f"Initial rotation: {angles}")

    return transformed_points, R, translation


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

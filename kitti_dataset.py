import numpy as np
import os
from matplotlib import pyplot as plt


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """
    Apply a 4×4 homogeneous transform to a point cloud that includes intensity.

    Args:
        points (np.ndarray):
            - Shape (N, 4), where columns = [x, y, z, intensity]
        transform (np.ndarray):
            - A 4×4 homogeneous transformation matrix.

    Returns:
        np.ndarray:
            - Shape (N, 4). The first three columns are the transformed (x, y, z),
              and the fourth column is the original intensity, carried over.
    """
    # 1) Basic sanity checks
    if transform.shape != (4, 4):
        raise ValueError(f"`transform` must be a 4×4 matrix, but got shape {transform.shape}")
    if points.ndim != 2 or points.shape[1] != 4:
        raise ValueError(f"`points` must have shape (N, 4), but got {points.shape}")

    # 2) Split into XYZ and intensity
    xyz = points[:, :3]  # (N, 3)
    intensity = points[:, 3:]  # (N, 1)

    # 3) Build homogeneous coordinates for XYZ: append a column of ones → (N, 4)
    N = xyz.shape[0]
    ones = np.ones((N, 1), dtype=xyz.dtype)
    pts_h = np.hstack((xyz, ones))  # → shape (N, 4)

    # 4) Apply the 4×4 transform: result is (N, 4)
    pts_h_trans = (transform @ pts_h.T).T  # → shape (N, 4)

    # 5) Extract transformed XYZ and re‐attach intensity
    xyz_transformed = pts_h_trans[:, :3]  # (N, 3)
    return np.hstack((xyz_transformed, intensity))  # (N, 4)


class KittiOdometryDataset:
    def __init__(self, data_path, sequence):
        """
        Initialize the KITTI Odometry dataset.

        Args:
            data_path (str): Path to the root directory of the KITTI dataset.
        """
        self.poses_in_lidar = None
        self.velo2cam = None
        self.data_path = data_path
        self.current_sequence = None
        self.calib = None
        self.poses = None
        self.load_sequence(sequence)

    def load_sequence(self, sequence):
        """
        Load a specific sequence from the dataset.

        Args:
            sequence (str): The sequence number (e.g., '00', '01', etc.).
        """
        self.current_sequence = sequence
        sequence_path = os.path.join(self.data_path, 'sequences', sequence)
        pose_path = os.path.join(self.data_path, 'poses', f'{sequence}.txt')

        # Load calibration data
        with open(os.path.join(sequence_path, 'calib.txt'), 'r') as f:
            self.calib = dict(line.strip().split(': ') for line in f if line.strip())
            for key in self.calib:
                self.calib[key] = np.array(list(map(float, self.calib[key].split()))).reshape((3, 4))

        self.velo2cam = np.vstack((self.calib['Tr'], [0, 0, 0, 1]))

        # Load poses
        self.poses = np.loadtxt(pose_path).reshape(-1, 3, 4)

        # Initialize poses in lidar frame
        self.poses_in_lidar = np.repeat(np.eye(4)[np.newaxis, :, :], self.poses.shape[0], axis=0)
        self.poses_in_lidar[:, :3, :4] = self.poses

        initial_pose = np.vstack((self.poses[0], [0, 0, 0, 1]))

        for i in range(self.poses.shape[0]):
            self.poses_in_lidar[i] = np.linalg.inv(self.velo2cam) @ \
                                     np.linalg.inv(initial_pose) @ \
                                     self.poses_in_lidar[i] @ \
                                     self.velo2cam

        timestamp_file = os.path.join(self.data_path, 'sequences', self.current_sequence,
                                      'timestamps.txt')

        with open(timestamp_file, 'r', encoding='utf-8') as f:
            self.timestamps = f.readlines()

    def get_lidar_data(self, frame, coordinate_system='lidar', drop_intensity=True):
        """
        Get LIDAR data for a specific frame in the current sequence.

        Args:
            frame (int): Frame number.
            coordinate_system (str): Coordinate system to return the data in.
        Returns:
            numpy.ndarray: Array of shape (N, 4) or (N, 3) containing x, y, z coordinates and reflection intensity.
            :param coordinate_system: lidar or world.
            :param drop_intensity: if True, the return value is array of shape (N, 3).
        """
        if self.current_sequence is None:
            raise ValueError("No sequence loaded. Call load_sequence() first.")

        lidar_filename = os.path.join(self.data_path, 'sequences', self.current_sequence,
                                      'velodyne', f'{frame:06d}.bin')
        result = None
        if coordinate_system == 'lidar':
            result = self._load_lidar_data(lidar_filename)
        elif coordinate_system == 'world':
            result = transform_points(self._load_lidar_data(lidar_filename), self.poses_in_lidar[frame])
        else:
            raise ValueError(f"Invalid coordinate system: {coordinate_system}")
        return result if not drop_intensity else result[:, :3]

    def _load_lidar_data(self, filename):
        """
        Load LIDAR point cloud from KITTI dataset binary file.

        Args:
            filename (str): Path to the binary file containing LIDAR data.

        Returns:
            numpy.ndarray: Array of shape (N, 4) containing x, y, z coordinates and reflection intensity.
        """
        lidar_data = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)  # [:, :3]
        return lidar_data

    def get_imu(self, frame):
        """
        Get IMU data for a specific frame in the current sequence.

        Args:
            frame (int): Frame number.

        Returns:
            numpy.ndarray: Array of shape (1, 9) containing IMU data
                           [gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z, vel_x, vel_y, vel_z].
        """

        imu_filename = os.path.join(self.data_path, 'sequences', self.current_sequence,
                                    'data', f'{frame:010d}.txt')

        imu_data = np.loadtxt(imu_filename)

        # Extract gyroscope, accelerometer, and velocity data
        vel = imu_data[8:11]  # [vf, vl, vu]
        accel = imu_data[11:14]  # [af, al, au]
        gyro = imu_data[20:23]  # [wf, wl, wu]

        imu = np.concatenate([gyro, accel, vel]).reshape(1, -1)

        return imu

    def get_dt(self, frame):
        """
        Get the time difference between the current frame and the previous frame.

        Args:
            frame (int): Current frame number.

        Returns:
            float: Time difference in seconds.
        """
        if self.current_sequence is None:
            raise ValueError("No sequence loaded. Call load_sequence() first.")

        if frame == 0:
            return 0.1  # First frame has no previous frame

        current_timestamp = self._get_timestamp(frame)
        previous_timestamp = self._get_timestamp(frame - 1)

        return current_timestamp - previous_timestamp

    def _get_timestamp(self, frame):
        """
        Get the timestamp for a specific frame.

        Args:
            frame (int): Frame number.

        Returns:
            float: Timestamp in seconds.
        """

        timestamp_str = self.timestamps[frame].strip()
        timestamp = float(timestamp_str.split(':')[-1])

        if not hasattr(self, 'start_time'):
            self.start_time = timestamp

        delta = timestamp - self.start_time
        return delta

    def plot_trajectory_from_pose(self, start_frame=0, end_frame=100):
        """
        Plot the trajectory from the current pose for the given number of frames.

        Args:
            num_frames (int): Number of frames to plot.
        """

        plt.title(f"Trajectory from sequence {self.current_sequence}")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        x, y, _ = self.get_x_y_z(start_frame, end_frame)
        plt.plot(x, y, '-b')
        plt.show()

    def get_x_y_z(self, start_frame, end_frames, step=1):
        """
        Get x and y coordinates for plotting the trajectory.

        Args:
            num_frames (int): Number of frames to consider.
            step (int): Step size for frame selection.

        Returns:
            tuple: Arrays of x and y coordinates.
        """
        return self.poses_in_lidar[start_frame:end_frames:step, 0, -1], \
            self.poses_in_lidar[start_frame:end_frames:step, 1, -1], \
            self.poses_in_lidar[start_frame:end_frames:step, 2, -1]

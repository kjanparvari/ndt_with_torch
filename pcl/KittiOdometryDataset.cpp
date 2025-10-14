#include "KittiOdometryDataset.h"
#include <iomanip>
#include <iostream>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <sstream>
#include <stdexcept>
#include <fstream>
#include <vector>

// Constructor to initialize and load sequence
KittiOdometryDataset::KittiOdometryDataset(const std::string &path,
                                           const std::string &sequence)
    : data_path(path), current_sequence(sequence)
{
  load_sequence(sequence);
}

// Method to load a specific sequence
void KittiOdometryDataset::load_sequence(const std::string &sequence)
{
  current_sequence = sequence;
  std::string sequence_path = data_path + "/sequences/" + sequence;
  std::string pose_path = data_path + "/poses/" + sequence + ".txt";

  // Load calibration data
  std::ifstream calib_file(sequence_path + "/calib.txt");

  if (!calib_file)
  {
    throw std::runtime_error("Failed to open calibration file!");
    return;
  }

  std::string line;

  while (std::getline(calib_file, line))
  {
    if (!line.empty())
    {
      std::istringstream ss(line);
      std::string key, value;
      std::getline(ss, key, ':');
      std::getline(ss, value);

      std::vector<float> calib_values;
      std::istringstream value_stream(value);
      float val;

      while (value_stream >> val)
      {
        calib_values.push_back(val);
      }
      if (calib_values.size() != 12)
      {
        throw std::runtime_error("Invalid calibration line: expected 12 values");
      }

      Eigen::Matrix<float, 3, 4> calib_matrix;

      for (int i = 0; i < 3; ++i)
      {
        for (int j = 0; j < 4; ++j)
        {
          calib_matrix(i, j) = calib_values[i * 4 + j];
        }
      }
      calib[key] = calib_matrix;
    }
  }

  // Construct velo2cam transformation matrix
  velo2cam.setIdentity();
  velo2cam.block<3, 4>(0, 0) = calib["Tr"];

  // Load poses
  std::ifstream pose_file(pose_path);

  if (!pose_file)
  {
    std::cerr << "Failed to open pose file!" << std::endl;

    return;
  }

  poses.clear();

  while (std::getline(pose_file, line))
  {
    Eigen::Matrix<float, 3, 4> pose;
    std::vector<float> pose_values;
    std::istringstream value_stream(line);
    float val;
    while (value_stream >> val)
    {
      pose_values.push_back(val);
    }
    for (int i = 0; i < 3; ++i)
    {
      for (int j = 0; j < 4; ++j)
      {
        pose(i, j) = pose_values[i * 4 + j];
      }
    }

    // pose << value, 0, 0, 0, value, 0, 0, 0, value, 0, 0, 0;
    poses.push_back(pose);
  }

  // Initialize poses_in_lidar
  poses_in_lidar.resize(poses.size(), Eigen::Matrix<float, 4, 4>::Identity());

  for (size_t i = 0; i < poses.size(); ++i)
  {
    // Copy the 3x4 pose into the top-left 3x4 block of a 4x4 matrix
    poses_in_lidar[i].block<3, 4>(0, 0) = poses[i];
    // Set the last row to [0, 0, 0, 1] (homogeneous transformation)
    poses_in_lidar[i].row(3) << 0, 0, 0, 1;
  }

  // Use the initial pose to transform all poses into the LIDAR frame
  if (poses.empty())
  {
    throw std::runtime_error("Pose file contains no poses");
  }
  Eigen::Matrix<float, 4, 4> initial_pose = poses_in_lidar[0];

  for (size_t i = 0; i < poses.size(); ++i)
  {
    // Transform poses into the lidar frame by applying the inverse of
    poses_in_lidar[i] = velo2cam.inverse() * initial_pose.inverse() *
                        poses_in_lidar[i] * velo2cam;
  }
  // Load timestamps
  std::ifstream timestamp_file(sequence_path + "/timestamps.txt");

  if (!timestamp_file)
  {
    std::cerr << "Failed to open timestamps file!" << std::endl;

    return;
  }

  timestamps.clear();

  while (std::getline(timestamp_file, line))
  {
    timestamps.push_back(line);
  }
}

// Method to get LIDAR data (without intensity if drop_intensity is true)
pcl::PointCloud<pcl::PointXYZI>::Ptr KittiOdometryDataset::get_lidar_data(
    int frame, const std::string &coordinate_system, float downsample_voxel)
{

  // Check if a sequence has been loaded
  if (current_sequence.empty())
  {
    throw std::runtime_error("No sequence loaded. Call load_sequence() first.");
  }
  if (frame < 0 || static_cast<size_t>(frame) >= poses_in_lidar.size())
  {
    throw std::out_of_range("Frame index out of range");
  }

  // Construct the lidar file path based on the frame number
  std::stringstream filename;
  filename << std::setw(6) << std::setfill('0') << frame << ".bin";
  std::string lidar_filename = data_path + "/sequences/" + current_sequence +
                               "/velodyne/" + filename.str();

  // Validate coordinate system
  if (coordinate_system != "world" && coordinate_system != "lidar")
  {
    throw std::invalid_argument("Invalid coordinate system: " +
                                coordinate_system);
  }

  // Load the lidar data from the file
  pcl::PointCloud<pcl::PointXYZI>::Ptr lidar_scan =
      _load_lidar_data(lidar_filename);

  // Downsample if voxel size is positive
  if (downsample_voxel > 0)
  {
    pcl::VoxelGrid<pcl::PointXYZI> voxel_filter;
    voxel_filter.setInputCloud(lidar_scan);
    voxel_filter.setLeafSize(downsample_voxel, downsample_voxel,
                             downsample_voxel);
    pcl::PointCloud<pcl::PointXYZI>::Ptr downsampled_cloud(
        new pcl::PointCloud<pcl::PointXYZI>);
    voxel_filter.filter(*downsampled_cloud); // Filter the cloud and store it
                                             // in a new pointer
    lidar_scan =
        downsampled_cloud; // Use downsampled cloud for further processing
  }

  // Transform or return the lidar data based on the coordinate system
  pcl::PointCloud<pcl::PointXYZI>::Ptr result(
      new pcl::PointCloud<pcl::PointXYZI>);

  if (coordinate_system == "lidar")
  {
    result = lidar_scan; // No transformation needed for lidar coordinates
  }
  else if (coordinate_system == "world")
  {
    // Transform the lidar scan to world coordinates
    Eigen::Matrix<float, 4, 4> pose = this->poses_in_lidar[frame];
    pcl::transformPointCloud(*lidar_scan, *result, pose, true);
  }

  return result;
}

// Private method to load lidar data from binary file
pcl::PointCloud<pcl::PointXYZI>::Ptr KittiOdometryDataset::_load_lidar_data(
    const std::string &filename)
{
  std::ifstream file(filename, std::ios::binary);

  if (!file)
  {
    throw std::runtime_error("Failed to open lidar data file: " + filename);
  }

  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(
      new pcl::PointCloud<pcl::PointXYZI>());
  pcl::PointXYZI point;

  while (file.read(reinterpret_cast<char *>(&point.x), sizeof(float)) &&
         file.read(reinterpret_cast<char *>(&point.y), sizeof(float)) &&
         file.read(reinterpret_cast<char *>(&point.z), sizeof(float)) &&
         file.read(reinterpret_cast<char *>(&point.intensity), sizeof(float)))
  {
    cloud->push_back(point);
  }

  file.close();

  return cloud;
}

// Function to create the Velodyne map
pcl::PointCloud<pcl::PointXYZI>::Ptr KittiOdometryDataset::create_velodyne_map(
    int start_frame, int end_frame, float map_downsample_voxel)
{
  // Accumulated points stored in Eigen matrix (3xN for x, y, z)
  pcl::PointCloud<pcl::PointXYZI>::Ptr velodyne_map(
      new pcl::PointCloud<pcl::PointXYZI>());

  for (int frame = start_frame; frame <= end_frame; ++frame)
  {
    // Get LIDAR data for the current frame
    pcl::PointCloud<pcl::PointXYZI>::Ptr lidar_data =
        this->get_lidar_data(frame, "world", map_downsample_voxel);
    (*velodyne_map) += (*lidar_data);
  }

  return velodyne_map;
}
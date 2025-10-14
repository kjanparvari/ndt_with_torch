#ifndef KITTI_ODOMETRY_DATASET_H
#define KITTI_ODOMETRY_DATASET_H

#include <Eigen/Dense>
#include <fstream>
#include <map>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <string>
#include <vector>

class KittiOdometryDataset {

 public:
  std::vector<Eigen::Matrix<float, 4, 4>> poses_in_lidar;

 public:
  // Constructor to initialize and load sequence
  KittiOdometryDataset(const std::string& path, const std::string& sequence);

  // Method to load a specific sequence
  void load_sequence(const std::string& sequence);

  // Method to get LIDAR data (without intensity if drop_intensity is true)
  pcl::PointCloud<pcl::PointXYZI>::Ptr get_lidar_data(
      int frame, const std::string& coordinate_system = "lidar",
      float downsample_voxel = -1.0f);

  pcl::PointCloud<pcl::PointXYZI>::Ptr create_velodyne_map(
      int start_frame, int end_frame, float map_downsample_voxel = .25f);

 private:
  // Private method to load lidar data from binary file
  pcl::PointCloud<pcl::PointXYZI>::Ptr _load_lidar_data(
      const std::string& filename);

 private:
  // Dataset-related members
  std::string data_path;
  std::string current_sequence;
  std::map<std::string, Eigen::Matrix<float, 3, 4>> calib;
  Eigen::Matrix<float, 4, 4> velo2cam;
  std::vector<Eigen::Matrix<float, 3, 4>> poses;
  std::vector<std::string> timestamps;
};

#endif  // KITTI_ODOMETRY_DATASET_H
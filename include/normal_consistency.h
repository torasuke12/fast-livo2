#pragma once

#include <Eigen/Core>
#include <vector>

namespace pcl { template <typename PointT> class PointCloud; struct PointXYZINormal; }

namespace fast_livo2::normal_consistency
{
struct Config
{
  bool enabled = false;
  int neighbors = 20;
  double max_neighbor_distance = 1.0;
  double max_planarity_ratio = 0.1;
  double min_spread_ratio = 0.1;
  double max_angle_deg = 30.0;
  void validate() const;
};

// Stable values recorded in geometry.csv, independent of correspondence status.
enum class Quality { NotComputed, Usable, InvalidPoint, InsufficientNeighbors,
                     NeighborDistance, Degenerate, NonPlanar, Narrow, NumericalFailure };
struct SourceNormal
{
  Eigen::Vector3d normal = Eigen::Vector3d::Zero(); // deskewed LiDAR frame
  Eigen::Vector3d eigenvalues = Eigen::Vector3d::Zero(); // centered PCA, ascending, m^2
  Quality quality = Quality::NotComputed;
  int neighbor_count = 0;
  double max_neighbor_distance = 0.0;
  bool usable() const { return quality == Quality::Usable; }
};
struct Comparison
{
  bool valid = false;
  bool mismatch = false; // independent of enabled; useful in an OFF capture
  double angle_deg = 0.0; // meaningful only when valid
};
SourceNormal fit(const std::vector<Eigen::Vector3d> &neighbors,
                 double max_distance, const Config &config);
std::vector<SourceNormal> estimate(const pcl::PointCloud<pcl::PointXYZINormal> &scan,
                                   const Config &config);
Comparison compare(const SourceNormal &source, const Eigen::Vector3d &map_normal,
                   const Eigen::Matrix3d &imu_to_world, const Eigen::Matrix3d &lidar_to_imu,
                   const Config &config);
} // namespace fast_livo2::normal_consistency

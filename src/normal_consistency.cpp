#include "normal_consistency.h"

#include <Eigen/Eigenvalues>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_types.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace fast_livo2::normal_consistency
{
void Config::validate() const
{
  if (neighbors < 3 || !std::isfinite(max_neighbor_distance) || max_neighbor_distance <= 0.0 ||
      !std::isfinite(max_planarity_ratio) || max_planarity_ratio <= 0.0 || max_planarity_ratio >= 1.0 ||
      !std::isfinite(min_spread_ratio) || min_spread_ratio <= 0.0 || min_spread_ratio >= 1.0 ||
      !std::isfinite(max_angle_deg) || max_angle_deg <= 0.0 || max_angle_deg > 90.0)
    throw std::invalid_argument("Invalid lio.normal_consistency: neighbors >= 3, radius > 0, "
                                "0 < PCA ratios < 1, 0 < angle <= 90; all thresholds finite");
}

SourceNormal fit(const std::vector<Eigen::Vector3d> &points, double max_distance, const Config &config)
{
  SourceNormal result;
  result.neighbor_count = static_cast<int>(points.size());
  result.max_neighbor_distance = max_distance;
  result.quality = Quality::InsufficientNeighbors;
  if (result.neighbor_count != config.neighbors) return result;
  result.quality = Quality::NumericalFailure;
  if (!std::isfinite(max_distance) || max_distance < 0.0) return result;
  for (const auto &point : points) if (!point.allFinite()) return result;
  result.quality = Quality::NeighborDistance;
  if (max_distance >= config.max_neighbor_distance) return result;
  Eigen::Vector3d center = Eigen::Vector3d::Zero();
  for (const auto &point : points) center += point;
  center /= points.size();
  Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
  for (const auto &point : points)
  {
    const Eigen::Vector3d centered = point - center;
    covariance.noalias() += centered * centered.transpose();
  }
  covariance /= points.size();
  result.quality = Quality::NumericalFailure;
  if (!covariance.allFinite()) return result;
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen(covariance);
  if (eigen.info() != Eigen::Success || !eigen.eigenvalues().allFinite()) return result;
  result.eigenvalues = eigen.eigenvalues();
  // A centered covariance is positive semidefinite; tolerate only roundoff.
  if (result.eigenvalues[0] < -1e-12 * std::max(1.0, result.eigenvalues[2])) return result;
  result.eigenvalues[0] = std::max(0.0, result.eigenvalues[0]);
  result.quality = Quality::Degenerate;
  if (result.eigenvalues[1] <= 1e-12 || result.eigenvalues[2] <= 1e-12) return result;
  result.quality = Quality::NonPlanar;
  if (result.eigenvalues[0] / result.eigenvalues[1] >= config.max_planarity_ratio) return result;
  result.quality = Quality::Narrow;
  if (result.eigenvalues[1] / result.eigenvalues[2] <= config.min_spread_ratio) return result;
  result.normal = eigen.eigenvectors().col(0).normalized();
  result.quality = result.normal.allFinite() ? Quality::Usable : Quality::NumericalFailure;
  return result;
}

std::vector<SourceNormal> estimate(const pcl::PointCloud<pcl::PointXYZINormal> &scan, const Config &config)
{
  // Build on XYZ only: normals/intensity in the input are unrelated to this PCA.
  pcl::PointCloud<pcl::PointXYZ>::Ptr finite(new pcl::PointCloud<pcl::PointXYZ>);
  finite->reserve(scan.size());
  for (const auto &point : scan)
    if (std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z))
      finite->emplace_back(point.x, point.y, point.z);
  pcl::KdTreeFLANN<pcl::PointXYZ> tree;
  if (!finite->empty()) tree.setInputCloud(finite);
  std::vector<SourceNormal> result(scan.size());
  std::vector<int> indices;
  std::vector<float> squared_distances;
  std::vector<Eigen::Vector3d> neighbors;
  neighbors.reserve(config.neighbors);
  for (size_t i = 0; i < scan.size(); ++i)
  {
    const auto &point = scan[i];
    if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z))
    {
      result[i].quality = Quality::InvalidPoint;
      continue;
    }
    const pcl::PointXYZ query(point.x, point.y, point.z);
    const int count = tree.nearestKSearch(query, config.neighbors, indices, squared_distances);
    neighbors.clear();
    for (int j = 0; j < count; ++j)
    {
      const auto &neighbor = (*finite)[indices[j]];
      neighbors.emplace_back(neighbor.x, neighbor.y, neighbor.z);
    }
    result[i] = fit(neighbors, count ? std::sqrt(squared_distances[count - 1]) : 0.0, config);
  }
  return result;
}

Comparison compare(const SourceNormal &source, const Eigen::Vector3d &map_normal,
                   const Eigen::Matrix3d &imu_to_world, const Eigen::Matrix3d &lidar_to_imu,
                   const Config &config)
{
  Comparison result;
  if (!source.usable()) return result;
  const Eigen::Vector3d world_normal = imu_to_world * lidar_to_imu * source.normal;
  if (!world_normal.allFinite() || !map_normal.allFinite() ||
      world_normal.norm() < 1e-12 || map_normal.norm() < 1e-12) return result;
  const double cosine = std::clamp(std::abs(world_normal.normalized().dot(map_normal.normalized())), 0.0, 1.0);
  constexpr double degrees = 180.0 / 3.14159265358979323846;
  result.valid = true;
  result.angle_deg = std::acos(cosine) * degrees;
  // Accept equality, allowing only floating-point roundoff at the boundary.
  result.mismatch = result.angle_deg > config.max_angle_deg + 1e-12;
  return result;
}
} // namespace fast_livo2::normal_consistency

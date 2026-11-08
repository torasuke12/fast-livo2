#include "voxel_map.h"

#include <algorithm>
#include <cmath>

namespace
{
void collectInitialPlaneNormals(const VoxelOctoTree *tree, size_t &count, M3D &normal_information)
{
  if (tree == nullptr) return;
  const auto &plane = *tree->plane_ptr_;
  if (plane.is_plane_)
  {
    // Do not count uncertain line-like/isotropic patches as independent plane
    // directions, even if the absolute minimum-eigenvalue test accepted them.
    if (plane.points_size_ > tree->points_size_threshold_ && plane.normal_.allFinite() &&
        plane.plane_var_.allFinite() && std::isfinite(plane.min_eigen_value_) &&
        std::isfinite(plane.mid_eigen_value_) &&
        plane.mid_eigen_value_ > std::max(1e-6f, 3.0f * std::max(0.0f, plane.min_eigen_value_)) &&
        plane.normal_.norm() > 1e-9)
    {
      const V3D normal = plane.normal_.normalized();
      normal_information += normal * normal.transpose();
      ++count;
    }
    // Matching also stops at a planar parent; do not double-count its children.
    return;
  }
  for (const auto *child : tree->leaves_) collectInitialPlaneNormals(child, count, normal_information);
}
} // namespace

void VoxelMapManager::resetVoxelInitialization()
{
  // No LIO/VIO update has run yet, so no visual references point into this map.
  for (auto &entry : voxel_map_) delete entry.second;
  voxel_map_.clear();
  initialization_points_.clear();
  const size_t resets = initialization_status_.resets + 1;
  initialization_status_ = VoxelMapInitializationStatus{};
  initialization_status_.resets = resets;
  initialization_start_time_ = -1.0;
  initialization_last_time_ = -1.0;
  initialization_log_time_ = -1.0;
}

void VoxelMapManager::reportVoxelInitialization(const double scan_time)
{
  if (!initialization_status_.ready && initialization_log_time_ >= 0.0 &&
      scan_time - initialization_log_time_ < 0.5) return;
  initialization_log_time_ = scan_time;
  const auto &s = initialization_status_;
  const auto &c = config_setting_.initialization_;
  RCLCPP_INFO(rclcpp::get_logger("voxel_map_init"),
              "[ Voxel Init ] %s: points=%zu/%d planes=%zu/%d normal_eigenvalues=[%.5f %.5f %.5f] "
              "min_required=%.5f duration=%.3f/%.3f s scans=%zu resets=%zu",
              s.ready ? "READY" : "WAITING", s.points, c.min_points, s.planes, c.min_planes,
              s.normal_eigenvalues.x(), s.normal_eigenvalues.y(), s.normal_eigenvalues.z(),
              c.min_normal_eigenvalue, s.duration_sec, c.min_duration_sec, s.scans, s.resets);
}

bool VoxelMapManager::TryInitializeVoxelMap(const double scan_time, std::vector<pointWithVar> *accepted_seed)
{
  if (accepted_seed) accepted_seed->clear();
  if (config_setting_.correspondence_mode_ != "voxel_plane") return false;
  if (initialization_status_.ready) return true;
  if (!std::isfinite(scan_time)) return false;
  if (scan_time == initialization_last_time_) return false; // Never count a repeated update twice.

  const auto &config = config_setting_.initialization_;
  if (initialization_start_time_ >= 0.0 &&
      (scan_time < initialization_last_time_ || scan_time - initialization_start_time_ > config.max_duration_sec))
  {
    RCLCPP_WARN(rclcpp::get_logger("voxel_map_init"),
                "[ Voxel Init ] Restarting accumulation after timeout/time reversal; "
                "points=%zu planes=%zu min_normal_eigenvalue=%.5f. "
                "No forced initialization; keep stationary with surfaces in multiple directions.",
                initialization_status_.points, initialization_status_.planes,
                initialization_status_.normal_eigenvalues.x());
    resetVoxelInitialization();
  }

  auto points = initialMapPoints();
  points.erase(std::remove_if(points.begin(), points.end(), [](const pointWithVar &point) {
    return !point.point_w.allFinite() || !point.var.allFinite();
  }), points.end());
  if (points.empty()) return false;

  const size_t capacity = static_cast<size_t>(config.max_points);
  if (initialization_points_.size() + points.size() > capacity && !initialization_points_.empty())
  {
    RCLCPP_WARN(rclcpp::get_logger("voxel_map_init"),
                "[ Voxel Init ] Accumulation point limit reached (%d); restarting without forcing readiness.",
                config.max_points);
    resetVoxelInitialization();
  }
  if (initialization_start_time_ < 0.0) initialization_start_time_ = scan_time;
  initialization_last_time_ = scan_time;

  // A single oversized scan is sampled evenly, instead of retaining just one
  // spatially sorted end of the cloud. The accumulation buffer remains bounded.
  const size_t count = std::min(points.size(), capacity);
  initialization_points_.reserve(std::min(capacity, initialization_points_.size() + count));
  for (size_t i = 0; i < count; ++i)
    initialization_points_.push_back(points[i * points.size() / count]);
  auto &status = initialization_status_;
  ++status.scans;
  status.points = initialization_points_.size();
  status.duration_sec = scan_time - initialization_start_time_;

  if (status.points >= static_cast<size_t>(config.min_points) && status.duration_sec >= config.min_duration_sec)
  {
    // Refit from the whole pending window. Incrementally freezing tiny early
    // planes would preserve exactly the sparse-seed error this gate prevents.
    for (auto &entry : voxel_map_) delete entry.second;
    voxel_map_.clear();
    buildVoxelMapFromPoints(initialization_points_);

    status.planes = 0;
    status.normal_eigenvalues.setZero();
    M3D information = M3D::Zero();
    for (const auto &entry : voxel_map_) collectInitialPlaneNormals(entry.second, status.planes, information);
    if (status.planes > 0)
    {
      Eigen::SelfAdjointEigenSolver<M3D> solver(information / static_cast<double>(status.planes));
      if (solver.info() == Eigen::Success && solver.eigenvalues().allFinite())
        status.normal_eigenvalues = solver.eigenvalues().cwiseMax(0.0);
    }
    // This checks normal-direction coverage, not full six-DoF observability.
    status.ready = status.planes >= static_cast<size_t>(config.min_planes) &&
                   status.normal_eigenvalues.x() >= config.min_normal_eigenvalue;
  }
  reportVoxelInitialization(scan_time);
  if (status.ready)
  {
    if (accepted_seed) accepted_seed->swap(initialization_points_);
    std::vector<pointWithVar>().swap(initialization_points_);
  }
  return status.ready;
}

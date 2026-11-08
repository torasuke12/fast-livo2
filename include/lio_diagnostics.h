#pragma once

#include <Eigen/Core>
#include <array>
#include <cstdint>
#include <fstream>
#include <functional>
#include <map>
#include <string>
#include <vector>
#include "normal_consistency.h"

namespace fast_livo2::diagnostics
{
struct Config
{
  bool enabled = false;
  std::string output_dir;
  bool detail_enabled = false;
  bool detail_final_only = false;
  bool geometry_enabled = false;
  double detail_start_sec = 170.0;
  double detail_end_sec = 220.0;
  bool xicp_enabled = false;
  std::string correspondence_mode = "fastlio_ikdtree";
  std::map<std::string, std::string> parameters;
  void validate() const;
};

enum class ScanStatus { Updated, NoMatches, Rejected, NoMap, NoUpdate };

enum class MatchStatus { InvalidPoint, InsufficientNeighbors, NeighborDistance, PlaneFit, Residual, Accepted, NormalMismatch,
                         NoVoxel, NoPlane, OutsideExtent, InvalidVariance };

// The selected voxel plane, or the rejected candidate that progressed furthest
// through matching (lowest normalized distance on ties). No fabricated 5-NN.
struct VoxelGeometry
{
  bool plane_valid = false, update_enabled = false;
  int plane_id = -1, layer = -1, support_points = 0;
  std::array<int64_t, 3> root_voxel{};
  Eigen::Vector3d voxel_center = Eigen::Vector3d::Zero(), eigenvalues = Eigen::Vector3d::Zero();
  Eigen::Matrix<double, 6, 6> plane_covariance = Eigen::Matrix<double, 6, 6>::Zero();
  Eigen::Matrix3d point_world_covariance = Eigen::Matrix3d::Zero(), point_body_covariance = Eigen::Matrix3d::Zero();
  double radius = 0.0, tangential_distance = 0.0, gate_distance = 0.0;
  double gate_plane_variance = 0.0, gate_point_variance = 0.0;
  double gate_variance = 0.0, probability = 0.0;
  double update_plane_variance = 0.0, update_point_variance = 0.0, update_variance = 0.0;
};

struct NormalStats
{
  double computation_ms = 0.0;
  int valid = 0, invalid = 0, comparisons = 0, rejected = 0;
};

struct GeometryPoint
{
  int match_index = -1;
  MatchStatus status = MatchStatus::InvalidPoint;
  Eigen::Vector3d lidar = Eigen::Vector3d::Zero(), world = Eigen::Vector3d::Zero();
  Eigen::Vector3d search_world = Eigen::Vector3d::Zero();
  int search_iteration = -2;
  double search_farthest_squared = 0.0;
  int neighbor_count = 0;
  std::array<Eigen::Vector3d, 5> neighbors{};
  Eigen::Vector3d normal = Eigen::Vector3d::Zero(), center = Eigen::Vector3d::Zero();
  double plane_d = 0.0, distance = 0.0;
  normal_consistency::SourceNormal source;
  normal_consistency::Comparison comparison;
  VoxelGeometry voxel;
};

struct GeometryState
{
  Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
  Eigen::Vector3d position = Eigen::Vector3d::Zero(), velocity = Eigen::Vector3d::Zero();
  Eigen::Vector3d bg = Eigen::Vector3d::Zero(), ba = Eigen::Vector3d::Zero(), gravity = Eigen::Vector3d::Zero();
  double inverse_exposure = 1.0;
};

// Observes the filter; never modifies its inputs or propagates logging failures.
// The caller guards calls with enabled() to avoid evaluating Eigen expressions
// (notably the baseline gain) when diagnostics are disabled or have failed.
class Logger
{
public:
  Logger(const Config &config, const std::string &default_directory,
         std::function<void(const std::string &)> warning = {});
  bool enabled() const { return config_.enabled && !failed_; }
  void disable(const std::string &reason) noexcept { fail(reason); }
  const std::string &runDirectory() const { return run_directory_; }
  void beginScan(double time, const Eigen::Ref<const Eigen::MatrixXd> &prior) noexcept;
  bool geometryEnabled() const;
  void normalStats(const NormalStats &stats) { normal_stats_ = stats; }
  void geometry(int index, const GeometryState &prior, const GeometryState &linearization,
                const GeometryState &after, const Eigen::Matrix3d &lidar_to_imu_rotation,
                const Eigen::Vector3d &lidar_to_imu_translation,
                const std::vector<GeometryPoint> &points) noexcept;
  void iteration(int index, bool final, const Eigen::MatrixXd &H,
                 const Eigen::VectorXd &inverse_variance, const Eigen::VectorXd &residual,
                 const Eigen::Ref<const Eigen::VectorXd> &prior_delta,
                 const Eigen::Ref<const Eigen::VectorXd> &solution,
                 const Eigen::Ref<const Eigen::MatrixXd> &P_iter, const Eigen::MatrixXd &gain,
                 const Eigen::MatrixXd &constraints, const Eigen::MatrixXd &target_gain,
                 double gain_constraint_error) noexcept;
  void endScan(ScanStatus status, int last_iteration,
               const Eigen::Ref<const Eigen::MatrixXd> &after) noexcept;

private:
  void open();
  void fail(const std::string &reason) noexcept;
  Config config_;
  std::string default_directory_, run_directory_;
  std::function<void(const std::string &)> warning_;
  bool failed_ = false, scan_open_ = false;
  uint64_t next_scan_id_ = 0, scan_id_ = 0;
  double time_ = 0.0, iteration_logging_ms_ = 0.0;
  double geometry_logging_ms_ = 0.0;
  NormalStats normal_stats_;
  int iterations_ = 0, committed_iteration_ = -2;
  Eigen::MatrixXd prior_;
  std::ofstream iterations_file_, scans_file_, details_file_, geometry_file_, geometry_states_file_;
};
} // namespace fast_livo2::diagnostics

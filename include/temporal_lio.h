#pragma once

// Fixed-lag LiDAR/IMU smoother with an optional conditional voxel-plane map.
// No ROS or PCL dependency.
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cstdint>
#include <array>
#include <map>
#include <limits>
#include <string>
#include <vector>

namespace temporal_lio {
using V3 = Eigen::Vector3d;
using M3 = Eigen::Matrix3d;
using V15 = Eigen::Matrix<double, 15, 1>;
using M15 = Eigen::Matrix<double, 15, 15>;
using M6 = Eigen::Matrix<double,6,6>;
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
M3 skew(const V3& x);
Eigen::Matrix<double,3,2> planeBasis(const V3& normal);
M3 expSO3(const V3& x);
V3 logSO3(const M3& r);
M3 rightJacobianInverse(const V3& x);

struct State {
  M3 rotation = M3::Identity();
  V3 position = V3::Zero(), velocity = V3::Zero();
  V3 gyro_bias = V3::Zero(), accel_bias = V3::Zero();
  State plus(const V15& dx) const;
  V15 minus(const State& origin) const;
  bool finite() const;
};
struct ImuSample {
  double time = 0;
  V3 gyro = V3::Zero(), accel = V3::Zero();
  // Preserve endpoint peaks when interval boundaries are interpolated. An
  // average below the sensor limit does not make a clipped sample reliable.
  V3 accel_max_abs = V3::Zero();
};
struct Return { V3 point = V3::Zero(); double time = 0; float intensity = 0; };
struct Config {
  double window_sec = 6.0, keyframe_sec = 0.5;
  int max_nodes = 16, iterations = 3, max_source_points = 1200;
  double voxel_size = 0.5, sample_size = 0.15;
  int min_plane_points = 6;
  double plane_ratio = 0.05, min_plane_spread = 0.01;
  double max_residual = 0.2, point_sigma = 0.03, huber_delta = 2.5;
  double max_imu_gap = 0.02;
  // SI units, after input scale conversion. Zero disables saturation handling.
  double accel_saturation = 0, accel_saturation_variance = 2500.;
  bool visual_degenerate_translation = false;
  double visual_degeneracy_ratio = .05;
  bool consistent_deskew = false;
  bool plane_uncertainty = false, persistent_map = false;
  int map_max_planes = 120000;
  bool structural_landmarks = false;
  int max_landmarks = 256;
  bool structure_enabled = false;
  double structure_sigma = 0.008726646259971648; // radians, 0.5 degrees per scan/direction
  double structure_max_angle = 0.13962634015954636; // radians, 8 degrees
  int structure_min_planes = 8;
  // Variances per IMU sample, matching imu.acc_cov / gyr_cov in this project.
  V3 gyro_variance = V3::Constant(0.1), accel_variance = V3::Constant(0.1);
  V3 gyro_bias_variance = V3::Constant(0.0001), accel_bias_variance = V3::Constant(0.0001);
  V3 gravity = V3(0, 0, -9.81);
  M3 lidar_to_imu_rotation = M3::Identity();
  V3 lidar_to_imu_translation = V3::Zero();
  void validate() const;
};

struct Preintegration {
  double dt = 0;
  int saturated_intervals = 0;
  M3 rotation = M3::Identity();
  V3 position = V3::Zero(), velocity = V3::Zero();
  V3 gyro_bias = V3::Zero(), accel_bias = V3::Zero();
  Eigen::Matrix<double, 9, 6> bias_jacobian = Eigen::Matrix<double, 9, 6>::Zero();
  M15 covariance = M15::Zero(), information = M15::Zero();
};
// Includes interpolated boundaries; short end extrapolation is explicit and bounded.
std::vector<ImuSample> interval(const std::vector<ImuSample>& samples, double begin,
                               double end, double max_gap);
Preintegration preintegrate(const std::vector<ImuSample>& samples, const State& state,
                           const Config& config);
State predict(const State& state, const Preintegration& delta, const V3& gravity);
V15 imuResidual(const State& a, const State& b, const Preintegration& delta, const V3& gravity);
Eigen::Matrix<double,15,30> imuJacobians(const State& a, const State& b,
                                       const Preintegration& delta, const V3& gravity);
std::vector<V3> deskew(const std::vector<Return>& points, const std::vector<ImuSample>& samples,
                       double end, const State& state, const Config& config);
std::vector<Eigen::Matrix<double,3,15>> deskewJacobians(const std::vector<Return>& points,
    const std::vector<ImuSample>& samples,double end,const State& state,const Config& config);

using VoxelKey = std::array<int, 3>;
struct Plane {
  V3 center = V3::Zero(), normal = V3::UnitZ();
  double variance = 0;
  int count = 0;
  // Extra normal/center uncertainty beyond variance's scatter/centroid term.
  // Order: normal xyz, center xyz. Includes their cross terms for a map plane.
  M6 additional_covariance = M6::Zero();
  double direction_information = 1.;
  std::vector<size_t> support;
  Eigen::Matrix<double,3,15> center_jacobian = Eigen::Matrix<double,3,15>::Zero();
  Eigen::Matrix<double,3,15> normal_jacobian = Eigen::Matrix<double,3,15>::Zero();
};
Plane refitPlane(const std::vector<V3>& points, const Plane& reference);
Plane worldPlane(const Plane& plane,const State& state,const M6& pose_covariance);
double planeAdditionalVariance(const Plane& plane,const V3& query);
VoxelKey voxelKey(const V3& point, double size);
std::map<VoxelKey, Plane> fitPlanes(const std::vector<V3>& points, const Config& config);
struct Node {
  uint64_t id = 0;
  double time = 0;
  bool keyframe = false;
  State state;
  M6 pose_covariance = M6::Identity();
  M3 covariance_rotation = M3::Identity();
  std::vector<Return> raw;
  std::vector<ImuSample> scan_imu;
  std::vector<V3> points;
  std::vector<Eigen::Matrix<double,3,15>> point_jacobians;
  // Bootstrap geometry in this node's IMU frame, already deskewed/accumulated
  // at acquisition. Keep it separate from the current scan's raw returns.
  std::vector<V3> initialization_reference;
  std::vector<size_t> source_indices;
  std::map<VoxelKey, Plane> planes;
  struct StructureMatch { Plane plane; V3 axis; double weight; VoxelKey key; };
  std::vector<StructureMatch> structure;
  struct LandmarkMatch { size_t source,slot,landmark; double weight; };
  std::vector<LandmarkMatch> landmark_matches;
  bool landmarks_assigned = false;
};
struct Report {
  std::string status;
  int nodes = 0, matches = 0, imu_edges = 0, marginalized = 0, iterations = 0;
  int map_matches = 0, map_planes = 0;
  int landmarks = 0;
  int saturated_imu_intervals = 0;
  int visual_rank = -1;
  double span = 0, cost_before = 0, cost_after = 0, imu_rms = 0, lidar_rms = 0;
  double max_imu_gap = 0, solve_ms = 0;
  V15 correction = V15::Zero();
  M15 covariance = M15::Identity();
  M15 information = M15::Zero();
};

// Image likelihood only, linearized in right rotation/world translation.
// Contains neither the frontend's state prior nor its posterior covariance.
struct PoseObservation {
  State origin;
  Matrix jacobian;
  Vector residual;
  struct Reference { uint64_t node; State origin; Matrix jacobian; };
  std::vector<Reference> references;
};
PoseObservation photometricObservation(const State& origin, const Matrix& jacobian,
                                      const Vector& residual, double pixel_variance,
                                      const std::vector<PoseObservation::Reference>& references = {});
Matrix pointAnchorJacobian(const State& anchor, const V3& world_point,
                           const Matrix& current_translation_jacobian);
PoseObservation weakTranslationObservation(const PoseObservation& observation,
                                           const M3& normal_information,double eigenvalue_ratio);

// Exposed for the linear-Gaussian marginalization regression test.
bool schurComplement(const Matrix& h, const Vector& b, int offset, int count,
                     Matrix& reduced_h, Vector& reduced_b);

class Window {
  friend struct DeskewDerivativeTest;
  friend struct MapWindowTest;
  friend struct LandmarkWindowTest;
 public:
  explicit Window(Config config);
  Report add(double time, const std::vector<Return>& points,
             const std::vector<ImuSample>& imu, const State& initial,
             const M15& initial_covariance,
             const std::vector<V3>& initialization_reference = {},
             bool retain_previous_for_image = false);
  const std::vector<Node>& nodes() const { return nodes_; }
  const State& state() const { return nodes_.back().state; }
  const Config& config() const { return config_; }
  Report addVisual(double time, const PoseObservation& observation);

 private:
  struct ImuEdge {
    uint64_t a, b;
    std::vector<ImuSample> samples;
    Preintegration delta;
  };
  struct Match { size_t source; Plane plane; double weight; VoxelKey key{}; };
  struct LidarEdge { uint64_t source, target; std::vector<Match> matches; };
  Config config_;
  std::vector<Node> nodes_;
  std::vector<ImuEdge> imu_edges_;
  std::vector<LidarEdge> lidar_edges_;
  struct VisualEdge { uint64_t node; PoseObservation observation; };
  std::vector<VisualEdge> visual_edges_;
  Matrix prior_jacobian_;
  Vector prior_residual_;
  std::vector<State> prior_origins_;
  struct Landmark { V3 normal,minimum,maximum; double distance; int structural_axis=-1; };
  // Plane distance is relative to this retained pose's estimated position.
  // Common world translation then has a constant null direction, even when
  // plane normals change after other states have been marginalized.
  uint64_t plane_anchor_id_ = 0;
  std::vector<Landmark> landmarks_;
  std::vector<Landmark> prior_landmark_origins_;
  uint64_t next_id_ = 0;
  double last_keyframe_time_ = -1;
  bool structure_initialized_ = false;
  M3 structure_axes_ = M3::Identity();
  void refreshStructure();
  void refreshLandmarks();
  static constexpr uint64_t map_target_ = std::numeric_limits<uint64_t>::max();
  std::map<VoxelKey,Plane> map_planes_;
  uint64_t last_visual_node_ = std::numeric_limits<uint64_t>::max();
  void insertMap(const Node& node);
  int index(uint64_t id) const;
  void refresh();
  void priorFactor(Matrix& jacobian, Vector& residual) const;
  void reanchorPlanes(uint64_t next_anchor);
  void linearize(Matrix& jacobian, Vector& residual, double& cost, int touching = -1) const;
  void marginalize(int index);
  int trim(bool retain_previous_for_image);
  void optimize(Report& report);
  double trialCost() const;
  void marginalCovariance(Report& report) const;
};
}  // namespace temporal_lio

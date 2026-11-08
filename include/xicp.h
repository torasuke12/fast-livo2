#pragma once

#include <Eigen/Core>
#include <array>
#include <string>

// X-ICP geometry and an equality-constrained iterated error-state Kalman update.
// Independent of ROS/PCL so frame, rank, and covariance behavior can be tested.
namespace fast_livo2::xicp
{
enum class Category { None = 0, Partial = 1, Full = 2 };

struct Config
{
  bool enabled = false;
  bool diagnostics = false;
  double contribution_min = 0.17364817766693041;  // cos(80 degrees)
  double contribution_strong = 0.7071067811865476; // cos(45 degrees)
  double enough_information = 250.0; // k1
  double insufficient_information = 180.0; // k2
  double minimal_information = 35.0; // k3
  double rank_relative_tolerance = 1e-6;
  void validate() const;
};

struct Direction
{
  Eigen::Vector3d axis = Eigen::Vector3d::Zero(); // LiDAR-local
  double eigenvalue = 0.0;
  double combined = 0.0;
  double strong = 0.0;
  int selected = 0;
  bool rank_fallback = false;
  Category category = Category::None;
};

struct Detection
{
  std::array<Direction, 6> directions; // rotation first, ascending eigenvalues
  Eigen::MatrixXd constraints;       // C_local, rows <= 6, columns = 6
  Eigen::MatrixXd target_gain;       // D, rows <= 6, columns = correspondences
};

// Points and unit normals are expressed at the LiDAR origin, NOT world origin.
// Geometry is unweighted, without the inertial prior, and rebuilt each iteration.
Detection detect(const Eigen::MatrixXd &points, const Eigen::MatrixXd &normals, const Config &config);

// Right IMU rotation error + world translation -> local LiDAR pose error.
// Includes the IMU-to-LiDAR lever arm; columns beyond pose are zero.
Eigen::MatrixXd localPoseJacobian(const Eigen::Matrix3d &world_from_imu,
                                 const Eigen::Matrix3d &imu_from_lidar,
                                 const Eigen::Vector3d &lidar_origin_in_imu, int state_size);

struct Update
{
  Eigen::VectorXd correction;
  Eigen::MatrixXd covariance;
  Eigen::MatrixXd gain;
  double constraint_error = 0.0;
};

// J contains the six pose columns of the observation Jacobian. All other
// observation columns are zero. Constraints apply to correction - prior_delta:
// C * (correction - prior_delta) = D * (residual - J * prior_delta.head(6)).
// D uses the SAME observation noise as residual, not a new pseudo-measurement.
// Throws on invalid/non-finite inputs or a numerically singular solve.
Update update(const Eigen::MatrixXd &prior_covariance, const Eigen::MatrixXd &jacobian,
              const Eigen::VectorXd &inverse_variance, const Eigen::VectorXd &residual,
              const Eigen::VectorXd &prior_delta, const Eigen::MatrixXd &constraints,
              const Eigen::MatrixXd &target_gain);
} // namespace fast_livo2::xicp

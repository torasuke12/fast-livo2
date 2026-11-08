#include "xicp.h"

#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace fast_livo2::xicp
{
void Config::validate() const
{
  if (!std::isfinite(contribution_min) || !std::isfinite(contribution_strong) ||
      !(0.0 < contribution_min && contribution_min <= contribution_strong && contribution_strong <= 1.0) ||
      !std::isfinite(enough_information) || !std::isfinite(insufficient_information) ||
      !std::isfinite(minimal_information) ||
      !(enough_information >= insufficient_information && insufficient_information > minimal_information &&
        minimal_information > 0.0) || !std::isfinite(rank_relative_tolerance) ||
      !(rank_relative_tolerance > 0.0 && rank_relative_tolerance < 1.0))
    throw std::invalid_argument("Invalid lio.xicp thresholds: require 0 < contribution_min <= strong <= 1, "
                                "k1 >= k2 > k3 > 0, and 0 < rank_relative_tolerance < 1");
}

Detection detect(const Eigen::MatrixXd &points, const Eigen::MatrixXd &normals, const Config &config)
{
  config.validate();
  if (points.cols() != 3 || normals.cols() != 3 || points.rows() != normals.rows() ||
      !points.allFinite() || !normals.allFinite())
    throw std::invalid_argument("X-ICP requires finite Nx3 points and normals");
  const int count = points.rows();
  Eigen::MatrixXd rotation(count, 3), capped_rotation(count, 3);
  for (int i = 0; i < count; ++i)
  {
    if (std::abs(normals.row(i).norm() - 1.0) > 1e-6)
      throw std::invalid_argument("X-ICP requires unit plane normals");
    const Eigen::Vector3d p = points.row(i), n = normals.row(i);
    const Eigen::Vector3d moment = p.cross(n);
    rotation.row(i) = moment;
    // Do not amplify points near the rotation axis. Translation information
    // remains useful even when this rotational moment is zero.
    capped_rotation.row(i) = moment / std::max(1.0, moment.norm());
  }
  if (!rotation.allFinite()) throw std::runtime_error("X-ICP moment overflow");

  Detection result;
  result.constraints = Eigen::MatrixXd::Zero(6, 6);
  result.target_gain = Eigen::MatrixXd::Zero(6, count);
  int constraint_count = 0;
  for (int block = 0; block < 2; ++block)
  {
    const Eigen::MatrixXd &raw = block == 0 ? rotation : normals;
    const Eigen::MatrixXd &contributions = block == 0 ? capped_rotation : normals;
    const Eigen::Matrix3d hessian = raw.transpose() * raw;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen(hessian);
    if (eigen.info() != Eigen::Success || !eigen.eigenvalues().allFinite())
      throw std::runtime_error("X-ICP geometric eigendecomposition failed");
    for (int axis = 0; axis < 3; ++axis)
    {
      Direction &direction = result.directions[3 * block + axis];
      direction.axis = eigen.eigenvectors().col(axis);
      direction.eigenvalue = eigen.eigenvalues()(axis);
      const Eigen::VectorXd contribution = (contributions * direction.axis).cwiseAbs();
      for (int i = 0; i < count; ++i)
      {
        if (contribution(i) >= config.contribution_min) direction.combined += contribution(i);
        if (contribution(i) >= config.contribution_strong) direction.strong += contribution(i);
      }
      if (direction.combined >= config.enough_information || direction.strong >= config.insufficient_information)
      {
        direction.category = Category::Full;
        continue;
      }
      if (direction.combined >= config.insufficient_information || direction.strong >= config.minimal_information)
      {
        direction.category = Category::Partial;
        // Figure 4: the combined-information branch takes precedence.
        const double gate = direction.combined >= config.insufficient_information
                                ? config.contribution_min : config.contribution_strong;
        Eigen::Matrix3d subset_hessian = Eigen::Matrix3d::Zero();
        std::vector<int> selected;
        for (int i = 0; i < count; ++i)
        {
          if (contribution(i) < gate) continue;
          subset_hessian.noalias() += raw.row(i).transpose() * raw.row(i);
          selected.push_back(i);
        }
        direction.selected = selected.size();
        // A full 3D auxiliary least-squares solve, NOT independent 1D division.
        // Rank-revealing SVD replaces the paper's LU/RIF numerical solver.
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(subset_hessian, Eigen::ComputeFullU | Eigen::ComputeFullV);
        svd.setThreshold(config.rank_relative_tolerance);
        const Eigen::Vector3d dual = svd.solve(direction.axis);
        if (!dual.allFinite() || (subset_hessian * dual - direction.axis).norm() > 1e-5)
        {
          direction.category = Category::None;
          direction.rank_fallback = true;
        }
        else
        {
          for (const int i : selected) result.target_gain(constraint_count, i) = raw.row(i).dot(dual);
        }
      }
      result.constraints.block<1, 3>(constraint_count, 3 * block) = direction.axis.transpose();
      ++constraint_count;
    }
  }
  result.constraints.conservativeResize(constraint_count, 6);
  result.target_gain.conservativeResize(constraint_count, count);
  return result;
}

Eigen::MatrixXd localPoseJacobian(const Eigen::Matrix3d &world_from_imu,
                                 const Eigen::Matrix3d &imu_from_lidar,
                                 const Eigen::Vector3d &lidar_origin_in_imu, const int state_size)
{
  if (state_size < 6 || !world_from_imu.allFinite() || !imu_from_lidar.allFinite() ||
      !lidar_origin_in_imu.allFinite()) throw std::invalid_argument("Invalid X-ICP frame transform");
  Eigen::Matrix3d lever;
  lever << 0, -lidar_origin_in_imu.z(), lidar_origin_in_imu.y(),
           lidar_origin_in_imu.z(), 0, -lidar_origin_in_imu.x(),
           -lidar_origin_in_imu.y(), lidar_origin_in_imu.x(), 0;
  Eigen::MatrixXd mapping = Eigen::MatrixXd::Zero(6, state_size);
  mapping.block<3, 3>(0, 0) = imu_from_lidar.transpose();
  mapping.block<3, 3>(3, 0) = -imu_from_lidar.transpose() * lever;
  mapping.block<3, 3>(3, 3) = (world_from_imu * imu_from_lidar).transpose();
  return mapping;
}
} // namespace fast_livo2::xicp

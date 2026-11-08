#pragma once
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <stdexcept>

namespace fast_livo2 {
using PoseCovariance = Eigen::Matrix<double,6,6>;

// Bound the marginal blocks by congruence, retaining pose cross-covariance
// and positive semidefiniteness. Rotation is a right/body perturbation;
// translation is a world-frame perturbation.
inline PoseCovariance boundedPoseCovariance(const PoseCovariance& input, double limit) {
  const PoseCovariance p = .5*(input+input.transpose());
  PoseCovariance scale = PoseCovariance::Identity();
  for(int k : {0,3}) {
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(p.block<3,3>(k,k));
    if(eig.info()!=Eigen::Success || !eig.eigenvalues().allFinite())
      throw std::runtime_error("Invalid pose covariance for point uncertainty");
    Eigen::Vector3d s;
    for(int i=0;i<3;++i) s[i]=eig.eigenvalues()[i]>limit ? std::sqrt(limit/eig.eigenvalues()[i]) : 1.;
    scale.block<3,3>(k,k)=eig.eigenvectors()*s.asDiagonal()*eig.eigenvectors().transpose();
  }
  return scale*p*scale.transpose();
}

inline Eigen::Matrix3d worldPointCovariance(const Eigen::Vector3d& point_imu,
    const Eigen::Matrix3d& world_from_imu, const Eigen::Matrix3d& imu_from_lidar,
    const Eigen::Matrix3d& measurement_covariance, const PoseCovariance& pose_covariance) {
  Eigen::Matrix3d cross;
  cross << 0,-point_imu.z(),point_imu.y(),point_imu.z(),0,-point_imu.x(),-point_imu.y(),point_imu.x(),0;
  Eigen::Matrix<double,3,6> j;
  j << -world_from_imu*cross, Eigen::Matrix3d::Identity();
  const Eigen::Matrix3d r=world_from_imu*imu_from_lidar;
  const Eigen::Matrix3d result=r*measurement_covariance*r.transpose()+j*pose_covariance*j.transpose();
  return .5*(result+result.transpose());
}
}

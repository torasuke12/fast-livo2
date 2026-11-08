#include "point_uncertainty.h"
#include <Eigen/Geometry>
#include <iostream>

int main() {
  using namespace fast_livo2;
  const Eigen::Vector3d p(3,-2,1);
  const Eigen::Matrix3d r=Eigen::AngleAxisd(.7,Eigen::Vector3d(1,2,3).normalized()).toRotationMatrix();
  const Eigen::Matrix3d extrinsic=Eigen::AngleAxisd(.3,Eigen::Vector3d::UnitY()).toRotationMatrix();
  const Eigen::Matrix3d noise=Eigen::Vector3d(.01,.02,.03).asDiagonal();
  const PoseCovariance root=PoseCovariance::Random();
  const PoseCovariance covariance=.01*root*root.transpose();
  const auto bounded=boundedPoseCovariance(covariance,.005);
  Eigen::SelfAdjointEigenSolver<PoseCovariance> eig(bounded);
  if(eig.eigenvalues().minCoeff() < -1e-12) return 1;
  for(int k:{0,3}) {
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> block(bounded.block<3,3>(k,k));
    if(block.eigenvalues().maxCoeff()>.005+1e-12) return 2;
  }
  Eigen::Matrix<double,3,6> numeric;
  const double eps=1e-6;
  for(int i=0;i<3;++i) {
    const Eigen::Vector3d axis=Eigen::Vector3d::Unit(i);
    numeric.col(i)=(r*Eigen::AngleAxisd(eps,axis)*p-r*Eigen::AngleAxisd(-eps,axis)*p)/(2*eps);
    numeric.col(3+i)=axis;
  }
  const Eigen::Matrix3d expected=r*extrinsic*noise*extrinsic.transpose()*r.transpose()+numeric*bounded*numeric.transpose();
  const auto result=worldPointCovariance(p,r,extrinsic,noise,bounded);
  if((result-expected).norm()>1e-8) return 3;
  const Eigen::Matrix3d q=Eigen::AngleAxisd(1.2,Eigen::Vector3d(3,1,2).normalized()).toRotationMatrix();
  PoseCovariance change=PoseCovariance::Identity();change.bottomRightCorner<3,3>()=q;
  const auto rotated=worldPointCovariance(p,q*r,extrinsic,noise,change*bounded*change.transpose());
  if((rotated-q*result*q.transpose()).norm()>1e-10) return 4;
  std::cout<<"Point covariance: finite differences, cross terms, PSD and world-frame equivariance OK\n";
}

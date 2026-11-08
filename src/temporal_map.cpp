#include "temporal_lio.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace temporal_lio {
double planeAdditionalVariance(const Plane& plane,const V3& query) {
  Eigen::Matrix<double,6,1> j;j<<query-plane.center,-plane.normal;
  const double variance=j.dot(plane.additional_covariance*j);
  if(!std::isfinite(variance) || variance<-1e-8) throw std::runtime_error("Invalid temporal plane uncertainty");
  return std::max(0.,variance);
}

Plane worldPlane(const Plane& body,const State& state,const M6& pose_covariance) {
  Plane world=body;
  world.center=state.rotation*body.center+state.position;world.normal=state.rotation*body.normal;
  M6 basis=M6::Zero();basis.topLeftCorner<3,3>()=state.rotation;basis.bottomRightCorner<3,3>()=state.rotation;
  M6 j=M6::Zero();j.topLeftCorner<3,3>()=-state.rotation*skew(body.normal);
  j.bottomLeftCorner<3,3>()=-state.rotation*skew(body.center);j.bottomRightCorner<3,3>().setIdentity();
  world.additional_covariance=basis*body.additional_covariance*basis.transpose()+j*pose_covariance*j.transpose();
  world.additional_covariance=.5*(world.additional_covariance+world.additional_covariance.transpose()).eval();
  world.support.clear();world.center_jacobian.setZero();world.normal_jacobian.setZero();
  return world;
}

void Window::insertMap(const Node& node) {
  M6 transport=M6::Identity();
  transport.topLeftCorner<3,3>()=rightJacobianInverse(logSO3(node.state.rotation.transpose()*node.covariance_rotation));
  const M6 covariance=transport*node.pose_covariance*transport.transpose();
  for(const auto& entry:node.planes) {
    Plane plane=worldPlane(entry.second,node.state,covariance);
    const auto key=voxelKey(plane.center,config_.voxel_size);
    auto found=map_planes_.find(key);
    if(found==map_planes_.end()) {map_planes_.emplace(key,std::move(plane));continue;}
    auto& old=found->second;
    // Mature map surfaces remain fixed. A new scan cannot move a wall merely
    // because its current pose has drifted. This is local scan-to-map matching,
    // without place recognition, loop factors, or optimization of old poses.
    if(old.count>=100) continue;
    double alignment=old.normal.dot(plane.normal);
    if(std::abs(alignment)<.985 || std::abs(old.normal.dot(plane.center-old.center))>.1) continue;
    if(alignment<0) {
      plane.normal=-plane.normal;
      M6 sign=M6::Identity();sign.topLeftCorner<3,3>()=-M3::Identity();
      plane.additional_covariance=sign*plane.additional_covariance*sign.transpose();
    }
    const int count=std::min(plane.count,100);
    const double alpha=static_cast<double>(count)/(old.count+count);
    old.center=(1-alpha)*old.center+alpha*plane.center;
    old.normal=((1-alpha)*old.normal+alpha*plane.normal).normalized();
    old.variance=(1-alpha)*old.variance+alpha*plane.variance;
    // Keep pose uncertainty instead of dividing a common acquisition error
    // by the number of returns. This is a conditional-map approximation;
    // cross-correlation between map planes and retained states is not stored.
    old.additional_covariance=(1-alpha)*old.additional_covariance+alpha*plane.additional_covariance;
    old.count+=count;
  }
  if(map_planes_.size()>static_cast<size_t>(config_.map_max_planes)) {
    std::vector<std::pair<double,VoxelKey>> distances;distances.reserve(map_planes_.size());
    for(const auto& entry:map_planes_) distances.emplace_back((entry.second.center-node.state.position).squaredNorm(),entry.first);
    const size_t remove=map_planes_.size()-config_.map_max_planes;
    std::partial_sort(distances.begin(),distances.begin()+remove,distances.end(),std::greater<>());
    for(size_t i=0;i<remove;++i) map_planes_.erase(distances[i].second);
  }
}
} // namespace temporal_lio

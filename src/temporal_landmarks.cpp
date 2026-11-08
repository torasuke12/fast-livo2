#include "temporal_lio.h"
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace temporal_lio {
Eigen::Matrix<double,3,2> planeBasis(const V3& normal) {
  Eigen::Matrix<double,3,2> basis;
  basis.col(0)=normal.unitOrthogonal();basis.col(1)=normal.cross(basis.col(0));return basis;
}
void Window::refreshLandmarks() {
  if(!config_.structural_landmarks) return;
  if(!structure_initialized_)
    throw std::runtime_error("Temporal structural landmarks: initial scan has no reliable Manhattan directions");
  const V3 anchor=nodes_.at(index(plane_anchor_id_)).state.position;
  for(auto& node:nodes_) {
    if(node.landmarks_assigned) continue;
    node.landmarks_assigned=true;
    // Membership is fixed for this scan. A factor entering the marginal prior
    // must never be silently assigned to another landmark on later iterations.
    std::map<VoxelKey,int> assignments;
    std::map<size_t,int> usage;
    auto angularBound=[&](const Plane& plane) {
      const V3 dots=structure_axes_.transpose()*node.state.rotation*plane.normal;
      const double angle=std::acos(std::clamp(dots.cwiseAbs().maxCoeff(),0.,1.));
      const M3 normal_covariance=plane.additional_covariance.topLeftCorner<3,3>();
      const double variance=Eigen::SelfAdjointEigenSolver<M3>(normal_covariance).eigenvalues().maxCoeff();
      return angle+2.*std::sqrt(std::max(0.,variance));
    };
    std::vector<size_t> order(node.source_indices.size());
    std::iota(order.begin(),order.end(),0);
    if(config_.plane_uncertainty) {
      // Reliable patches establish directions before weak patches try to
      // associate with them. A thin six-return patch cannot invent a precise
      // Manhattan direction merely because its noisy mean is near an axis.
      std::vector<double> priority(order.size(),1e100);
      for(size_t slot:order) {
        const auto it=node.planes.find(voxelKey(node.points[node.source_indices[slot]],config_.voxel_size));
        if(it!=node.planes.end()) priority[slot]=angularBound(it->second);
      }
      std::stable_sort(order.begin(),order.end(),[&](size_t a,size_t b){return priority[a]<priority[b];});
    }
    for(size_t slot:order) {
      const size_t source=node.source_indices[slot];
      const VoxelKey key=voxelKey(node.points[source],config_.voxel_size);
      const auto plane_it=node.planes.find(key);
      if(plane_it==node.planes.end()) continue;
      const Plane& plane=plane_it->second;
      auto assignment=assignments.find(key);
      if(assignment==assignments.end()) {
        const V3 normal=node.state.rotation*plane.normal;
        const V3 dots=structure_axes_.transpose()*normal;
        Eigen::Index axis;
        const bool near_axis=dots.cwiseAbs().maxCoeff(&axis)>=std::cos(config_.structure_max_angle);
        const bool structural=near_axis && (!config_.plane_uncertainty || angularBound(plane)<config_.structure_max_angle);
        const V3 canonical=structural ? V3(structure_axes_.col(axis)) : normal;
        const V3 center=node.state.rotation*plane.center+node.state.position;
        const V3 relative_center=node.state.rotation*plane.center+(node.state.position-anchor);
        const double distance=canonical.dot(relative_center);
        int best=-1;double score=std::min(.12,config_.max_residual);
        for(size_t i=0;i<landmarks_.size();++i) {
          const auto& landmark=landmarks_[i];
          // A free plane that rotates close to an axis must not silently
          // absorb structural observations and bypass their direction prior.
          if(structural && landmark.structural_axis!=axis) continue;
          if(!structural && landmark.structural_axis>=0 &&
             (!near_axis || landmark.structural_axis!=axis)) continue;
          // An uncertain patch may observe an already established direction,
          // or remain a free plane. It never creates a new direction prior.
          if(std::abs(landmark.normal.dot(canonical))<std::cos(config_.structure_max_angle)) continue;
          // Local plane association only. There is no place recognition,
          // return-to-start constraint, or revision of already emitted poses.
          // Include the 2 m boundary regardless of world-origin roundoff.
          // The nanometre guard is far below measurement resolution; without
          // it, translating an identical scene could split one floor in two.
          constexpr double support_margin=2.+1e-9;
          if(((center-landmark.minimum).array()<-support_margin).any() ||
             ((center-landmark.maximum).array()>support_margin).any()) continue;
          const double separation=std::abs(landmark.normal.dot(relative_center)-landmark.distance);
          if(separation<score) {score=separation;best=i;}
        }
        if(best<0 && landmarks_.size()<static_cast<size_t>(config_.max_landmarks)) {
          best=landmarks_.size();landmarks_.push_back({canonical,center,center,distance,structural ? static_cast<int>(axis) : -1});
          // A new general plane has a flat prior: its normal and distance are
          // inferred from the raw returns and correlated with acquisition pose.
          // A structural direction contributes one soft prior, not one per scan.
          const int rows=prior_jacobian_.rows(),cols=prior_jacobian_.cols();
          const int added=structural ? 2 : 0;
          Matrix extended=Matrix::Zero(rows+added,cols+3);
          extended.topLeftCorner(rows,cols)=prior_jacobian_;
          if(structural) extended.block<2,2>(rows,cols)=Eigen::Matrix2d::Identity()/config_.structure_sigma;
          prior_jacobian_=std::move(extended);
          prior_residual_.conservativeResize(rows+added);
          if(added) prior_residual_.tail(added).setZero();
          prior_landmark_origins_.push_back(landmarks_.back());
        }
        assignment=assignments.emplace(key,best).first;
        if(best>=0) {
          auto& landmark=landmarks_[best];
          landmark.minimum=landmark.minimum.cwiseMin(center);
          landmark.maximum=landmark.maximum.cwiseMax(center);
        }
      }
      if(assignment->second<0) continue;
      const size_t landmark=assignment->second;
      // Local scatter also covers departures from an ideal structural plane.
      const double weight=1./(config_.point_sigma*config_.point_sigma+plane.variance);
      node.landmark_matches.push_back({source,slot,landmark,weight});++usage[landmark];
    }
    // Bound scan-wise information from correlated returns on one surface.
    for(auto& match:node.landmark_matches) match.weight*=std::min(1.,50./usage.at(match.landmark));
  }
}
} // namespace temporal_lio

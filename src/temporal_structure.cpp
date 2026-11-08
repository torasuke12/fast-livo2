#include "temporal_lio.h"
#include <cmath>

namespace temporal_lio {
void Window::refreshStructure() {
  if(!config_.structure_enabled) return;
  if(!structure_initialized_) {
    // Learn heading only from the first available local scan. No position,
    // floor height, trajectory endpoint, or future measurement enters here.
    const auto& first=nodes_.front();
    const V3 up=-config_.gravity.normalized(),x=up.unitOrthogonal(),y=up.cross(x);
    double sine=0,cosine=0;int count=0;
    for(const auto& entry:first.planes) {
      const V3 n=first.state.rotation*entry.second.normal;
      if(std::abs(n.dot(up))>std::sin(config_.structure_max_angle)) continue;
      const double angle=std::atan2(n.dot(y),n.dot(x));
      sine+=std::sin(4*angle);cosine+=std::cos(4*angle);++count;
    }
    if(count<3*config_.structure_min_planes || std::hypot(sine,cosine)<.85*count) return;
    const double heading=std::atan2(sine,cosine)/4;
    structure_axes_.col(0)=x*std::cos(heading)+y*std::sin(heading);
    structure_axes_.col(1)=up.cross(structure_axes_.col(0));structure_axes_.col(2)=up;
    structure_initialized_=true;
  }
  // The raw point-to-landmark factors already observe orientation in this
  // mode. Do not count normals extracted from the same returns a second time.
  if(config_.structural_landmarks) return;
  for(auto& node:nodes_) {
    node.structure.clear();
    std::array<int,3> count{};
    std::array<double,3> information{};
    std::vector<int> axes;
    for(const auto& entry:node.planes) {
      const auto& plane=entry.second;
      const V3 n=node.state.rotation*plane.normal;
      const V3 dots=structure_axes_.transpose()*n;
      Eigen::Index axis;const double alignment=dots.cwiseAbs().maxCoeff(&axis);
      if(alignment<std::cos(config_.structure_max_angle)) continue;
      const V3 direction=structure_axes_.col(axis)*(dots[axis]<0 ? -1. : 1.);
      node.structure.push_back({plane,direction,0,entry.first});axes.push_back(axis);++count[axis];
      information[axis]+=config_.plane_uncertainty ? plane.direction_information : 1.;
    }
    // Correlated surface patches do not multiply orientation confidence. Each
    // observed direction has at most one scan's configured information.
    for(size_t i=0;i<node.structure.size();++i) {
      const int n=count[axes[i]];
      if(n>=config_.structure_min_planes)
        node.structure[i].weight=(config_.plane_uncertainty ? node.structure[i].plane.direction_information : 1.)
          /(config_.structure_sigma*config_.structure_sigma*information[axes[i]]);
    }
  }
}
} // namespace temporal_lio

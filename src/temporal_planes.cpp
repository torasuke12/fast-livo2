#include "temporal_lio.h"
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace temporal_lio {
VoxelKey voxelKey(const V3& p, double size) {
  VoxelKey key;
  for(int k=0;k<3;++k) {
    double v=std::floor(p[k]/size);
    if(!std::isfinite(v) || std::abs(v)>1e8) throw std::runtime_error("Temporal LIO: invalid voxel coordinate");
    key[k]=static_cast<int>(v);
  }
  return key;
}
std::map<VoxelKey, Plane> fitPlanes(const std::vector<V3>& points, const Config& c) {
  struct Moments { int n=0; V3 mean=V3::Zero(); M3 sum=M3::Zero(); std::vector<size_t> support; };
  std::map<VoxelKey, Moments> bins;
  for(size_t i=0;i<points.size();++i) {
    const V3& p=points[i];
    auto& b=bins[voxelKey(p,c.voxel_size)]; ++b.n;
    if(c.consistent_deskew) b.support.push_back(i);
    V3 d=p-b.mean; b.mean+=d/b.n; b.sum+=d*(p-b.mean).transpose();
  }
  std::map<VoxelKey, Plane> planes;
  for(const auto& entry:bins) {
    const auto& b=entry.second;
    if(b.n<c.min_plane_points) continue;
    Eigen::SelfAdjointEigenSolver<M3> eig(b.sum/b.n);
    if(eig.info()!=Eigen::Success) continue;
    const V3 e=eig.eigenvalues();
    if(e[1]<1e-5 || e[1]<c.min_plane_spread*e[2] || e[0]>c.plane_ratio*e[1]) continue;
    Plane p; p.center=b.mean; p.normal=eig.eigenvectors().col(0); p.count=b.n;
    p.support=b.support;
    // Surface scatter and finite support enter observation variance. Acquisition
    // pose uncertainty is represented by the target node, never inserted again.
    p.variance=std::max(0.0,e[0])*(1.0+1.0/b.n)+c.point_sigma*c.point_sigma/b.n;
    for(int k=1;k<3;++k) {
      const V3 axis=eig.eigenvectors().col(k);
      p.additional_covariance.topLeftCorner<3,3>()+=
        c.point_sigma*c.point_sigma/b.n*(e[k]+e[0])/std::pow(e[k]-e[0],2)*axis*axis.transpose();
    }
    p.direction_information=2./std::max(1e-12,p.additional_covariance.topLeftCorner<3,3>().trace());
    planes.emplace(entry.first,p);
  }
  return planes;
}

Plane refitPlane(const std::vector<V3>& points,const Plane& reference) {
  if(reference.support.empty()) return reference;
  Plane result=reference;result.center.setZero();
  for(size_t i:reference.support) result.center+=points.at(i);
  result.center/=reference.support.size();
  M3 covariance=M3::Zero();
  for(size_t i:reference.support) {const V3 d=points[i]-result.center;covariance+=d*d.transpose();}
  Eigen::SelfAdjointEigenSolver<M3> eig(covariance/reference.support.size());
  if(eig.info()!=Eigen::Success) throw std::runtime_error("Temporal plane refit failed");
  result.normal=eig.eigenvectors().col(0);
  if(result.normal.dot(reference.normal)<0) result.normal=-result.normal;
  return result;
}

void Window::refresh() {
  for(auto& n:nodes_) {
    n.points=deskew(n.raw,n.scan_imu,n.time,n.state,config_);
    n.planes=fitPlanes(n.initialization_reference.empty() ? n.points : n.initialization_reference,config_);
    if(config_.consistent_deskew && !config_.structural_landmarks) {
      n.point_jacobians=deskewJacobians(n.raw,n.scan_imu,n.time,n.state,config_);
      if(n.initialization_reference.empty()) for(auto& entry:n.planes) {
        auto& plane=entry.second;
        M3 covariance=M3::Zero();
        for(size_t i:plane.support) {
          const V3 d=n.points[i]-plane.center;covariance+=d*d.transpose();
          plane.center_jacobian+=n.point_jacobians[i]/plane.count;
        }
        Eigen::SelfAdjointEigenSolver<M3> eig(covariance/plane.count);
        M3 sensitivity=M3::Zero();
        for(int k=1;k<3;++k) sensitivity+=eig.eigenvectors().col(k)*eig.eigenvectors().col(k).transpose()
            /(eig.eigenvalues()[0]-eig.eigenvalues()[k]);
        for(int k:{0,1,2,6,7,8,9,10,11,12,13,14}) {
          V3 dc_n=V3::Zero();
          for(size_t i:plane.support) {
            const V3 d=n.points[i]-plane.center;
            const V3 j=n.point_jacobians[i].col(k)-plane.center_jacobian.col(k);
            dc_n+=(j*d.dot(plane.normal)+d*j.dot(plane.normal))/plane.count;
          }
          plane.normal_jacobian.col(k)=sensitivity*dc_n;
        }
      }
    }
    // Keep actual measured returns, including their acquisition timestamps.
    // Voxel centroids would mix returns from different times.
    if(n.source_indices.empty() && config_.structural_landmarks) {
      // General plane landmarks need spread in two tangent directions. Keep
      // six actual returns per fitted patch (and their individual timestamps),
      // rather than leaving a new 3-DOF plane supported by a single return.
      const auto raw_planes=n.initialization_reference.empty() ? n.planes : fitPlanes(n.points,config_);
      std::vector<const Plane*> candidates;
      for(const auto& entry:raw_planes) candidates.push_back(&entry.second);
      const size_t count=std::min(candidates.size(),static_cast<size_t>(config_.max_source_points/6));
      for(size_t k=0;k<count;++k) {
        const auto& support=candidates[k*candidates.size()/count]->support;
        std::array<size_t,6> selected;V3 mean=V3::Zero();
        for(size_t j=0;j<6;++j) {selected[j]=support[j*support.size()/6];mean+=n.points[selected[j]]/6.;}
        M3 scatter=M3::Zero();
        for(size_t i:selected) {const V3 d=n.points[i]-mean;scatter+=d*d.transpose();}
        if(Eigen::SelfAdjointEigenSolver<M3>(scatter/6.).eigenvalues()[1]<1e-6) continue;
        n.source_indices.insert(n.source_indices.end(),selected.begin(),selected.end());
      }
    } else if(n.source_indices.empty()) {
      std::map<VoxelKey,size_t> selected;
      for(size_t i=0;i<n.points.size();++i) selected.emplace(voxelKey(n.points[i],config_.sample_size),i);
      std::vector<size_t> all;
      for(const auto& kv:selected) all.push_back(kv.second);
      const size_t count=std::min(all.size(),static_cast<size_t>(config_.max_source_points));
      for(size_t k=0;k<count;++k) n.source_indices.push_back(all[k*all.size()/count]);
    }
    if(config_.consistent_deskew && config_.structural_landmarks) {
      std::vector<Return> selected;
      // Preserve the exact first interpolation boundary used by the full
      // deskew, even if the earliest return was not selected as a source.
      selected.push_back(*std::min_element(n.raw.begin(),n.raw.end(),[](const Return& a,const Return& b){return a.time<b.time;}));
      for(size_t i:n.source_indices) selected.push_back(n.raw[i]);
      n.point_jacobians=deskewJacobians(selected,n.scan_imu,n.time,n.state,config_);
      n.point_jacobians.erase(n.point_jacobians.begin());
    }
  }
  for(auto& edge:imu_edges_)
    edge.delta=preintegrate(edge.samples,nodes_.at(index(edge.a)).state,config_);
  for(auto& edge:lidar_edges_) {
    const auto& source=nodes_.at(index(edge.source));
    const auto associate=[&](uint64_t target_id) {
    edge.target=target_id;edge.matches.clear();
    const bool fixed=target_id==map_target_;
    const auto* target=fixed ? nullptr : &nodes_.at(index(target_id));
    const auto& planes=fixed ? map_planes_ : target->planes;
    const M3 r=fixed ? source.state.rotation : M3(target->state.rotation.transpose()*source.state.rotation);
    const V3 t=fixed ? source.state.position : V3(target->state.rotation.transpose()*(source.state.position-target->state.position));
    std::map<VoxelKey,int> usage;
    std::vector<VoxelKey> keys;
    for(size_t i:source.source_indices) {
      const V3 q=r*source.points[i]+t;
      const auto key=voxelKey(q,config_.voxel_size);
      const Plane* best=nullptr; VoxelKey best_key{};
      double best_score=1e100;
      for(int x=-1;x<=1;++x) for(int y=-1;y<=1;++y) for(int z=-1;z<=1;++z) {
        VoxelKey k{key[0]+x,key[1]+y,key[2]+z}; auto it=planes.find(k);
        if(it==planes.end()) continue;
        const auto& p=it->second; const V3 d=q-p.center;
        const double residual=std::abs(p.normal.dot(d));
        if(residual>config_.max_residual || d.squaredNorm()>2.25*config_.voxel_size*config_.voxel_size) continue;
        const double score=residual*residual+0.01*d.squaredNorm();
        if(score<best_score) {best=&p;best_key=k;best_score=score;}
      }
      if(!best) continue;
      const double extra=config_.plane_uncertainty || fixed ? planeAdditionalVariance(*best,q) : 0.;
      edge.matches.push_back({i,*best,1.0/(config_.point_sigma*config_.point_sigma+best->variance+extra),best_key});
      keys.push_back(best_key); ++usage[best_key];
    }
    // A conservative support cap prevents a dense patch from dominating. This
    // is not a complete shared-landmark covariance model (see documentation).
    for(size_t i=0;i<edge.matches.size();++i)
      edge.matches[i].weight*=std::min(1.0,6.0/usage.at(keys[i]));
    };
    if(config_.persistent_map && !map_planes_.empty()) {
      associate(map_target_);
      // Newly visible surfaces may not yet be in the finalized map. Use one
      // retained reference instead, without counting the source scan twice.
      if(edge.matches.size()>=30) continue;
      auto target=std::find_if(nodes_.rbegin(),nodes_.rend(),[&](const Node& n){return n.keyframe && n.id<source.id;});
      if(target!=nodes_.rend()) associate(target->id);
    } else associate(edge.target);
  }
  refreshStructure();
  refreshLandmarks();
}
} // namespace temporal_lio

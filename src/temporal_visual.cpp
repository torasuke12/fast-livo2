#include "temporal_lio.h"
#include <Eigen/QR>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <set>

namespace temporal_lio {
namespace {
Matrix imageJacobian(const PoseObservation& observation) {
  Matrix result(observation.jacobian.rows(),6*(1+observation.references.size()));
  result.leftCols(6)=observation.jacobian;
  for(size_t i=0;i<observation.references.size();++i)
    result.middleCols(6*(i+1),6)=observation.references[i].jacobian;
  return result;
}
void assignImageJacobian(PoseObservation& observation,const Matrix& matrix) {
  observation.jacobian=matrix.leftCols(6);
  for(size_t i=0;i<observation.references.size();++i)
    observation.references[i].jacobian=matrix.middleCols(6*(i+1),6);
}
}

Matrix pointAnchorJacobian(const State& anchor,const V3& world_point,const Matrix& translation) {
  if(!anchor.finite() || !world_point.allFinite() || translation.cols()!=3 || !translation.allFinite())
    throw std::invalid_argument("Invalid visual point anchor");
  const V3 body=anchor.rotation.transpose()*(world_point-anchor.position);
  Matrix result(translation.rows(),6);
  result.leftCols(3)=translation*anchor.rotation*skew(body);
  result.rightCols(3)=-translation;
  return result;
}

PoseObservation photometricObservation(const State& origin,const Matrix& j,
                                      const Vector& r,double variance,
                                      const std::vector<PoseObservation::Reference>& references) {
  if(!origin.finite() || (j.cols()!=6 && j.cols()!=7) || j.rows()!=r.size() ||
     !j.allFinite() || !r.allFinite() || !std::isfinite(variance) || variance<=0)
    throw std::invalid_argument("Temporal visual: invalid image likelihood");
  std::set<uint64_t> ids;
  for(const auto& reference:references)
    if(!reference.origin.finite() || reference.jacobian.cols()!=6 || reference.jacobian.rows()!=r.size() ||
       !reference.jacobian.allFinite() || !ids.insert(reference.node).second)
      throw std::invalid_argument("Temporal visual: invalid reference likelihood");
  PoseObservation out;out.origin=origin;out.references=references;
  if(j.rows()==0) {out.jacobian.resize(0,6);return out;}
  // Eliminate only the per-image exposure. Retain all current/reference pose
  // columns together so their cross-information survives compression.
  const bool exposure=j.cols()==7 && j.col(6).squaredNorm()>1e-20;
  const int offset=exposure ? 1 : 0,pose_columns=6*(1+references.size());
  Matrix augmented(j.rows(),offset+pose_columns+1);
  if(exposure) augmented.col(0)=j.col(6);
  augmented.middleCols(offset,6)=j.leftCols(6);
  for(size_t i=0;i<references.size();++i)
    augmented.middleCols(offset+6*(i+1),6)=references[i].jacobian;
  augmented.col(augmented.cols()-1)=r;
  augmented/=std::sqrt(variance);
  Eigen::HouseholderQR<Matrix> qr(augmented);
  const int rows=std::min<int>(augmented.rows(),augmented.cols());
  const Matrix root=qr.matrixQR().topRows(rows).triangularView<Eigen::Upper>();
  assignImageJacobian(out,root.block(offset,offset,rows-offset,pose_columns));
  out.residual=root.block(offset,augmented.cols()-1,rows-offset,1);
  return out;
}

PoseObservation weakTranslationObservation(const PoseObservation& observation,
                                           const M3& information,double ratio) {
  if(!information.allFinite() || !std::isfinite(ratio) || ratio<=0 || ratio>=1)
    throw std::invalid_argument("Invalid visual translation geometry");
  Eigen::SelfAdjointEigenSolver<M3> eig(.5*(information+information.transpose()));
  if(eig.info()!=Eigen::Success || eig.eigenvalues().minCoeff()<-1e-8)
    throw std::invalid_argument("Invalid visual translation information");
  int weak=0;
  const double threshold=std::max(1e-12,ratio*eig.eigenvalues().maxCoeff());
  while(weak<3 && eig.eigenvalues()[weak]<threshold) ++weak;
  PoseObservation out;out.origin=observation.origin;out.jacobian.resize(0,6);
  if(!weak || observation.jacobian.rows()==0) return out;
  out.references=observation.references;
  const int count=1+out.references.size(),keep_columns=weak*count;
  Matrix keep=Matrix::Zero(6*count,keep_columns),drop=Matrix::Zero(6*count,(6-weak)*count);
  for(int k=0;k<count;++k) {
    keep.block(6*k+3,weak*k,3,weak)=eig.eigenvectors().leftCols(weak);
    drop.block(6*k,(6-weak)*k,3,3).setIdentity();
    if(weak<3) drop.block(6*k+3,(6-weak)*k+3,3,3-weak)=eig.eigenvectors().rightCols(3-weak);
  }
  // Marginalize nuisance components for every participating pose. Keeping
  // both endpoints preserves the global-translation gauge of an image pair.
  const Matrix joint=imageJacobian(observation),nuisance=joint*drop;
  Eigen::ColPivHouseholderQR<Matrix> qr(nuisance);qr.setThreshold(1e-9);
  Matrix remaining(joint.rows(),keep_columns+1);
  remaining.leftCols(keep_columns)=joint*keep;remaining.col(keep_columns)=observation.residual;
  const Matrix projected=qr.householderQ().adjoint()*remaining;
  const int rows=projected.rows()-qr.rank();
  assignImageJacobian(out,projected.bottomRows(rows).leftCols(keep_columns)*keep.transpose());
  out.residual=projected.bottomRows(rows).col(keep_columns);
  return out;
}

Report Window::addVisual(double time,const PoseObservation& observation) {
  if(nodes_.empty() || !std::isfinite(time) || std::abs(time-nodes_.back().time)>1e-6 ||
     !observation.origin.finite() || observation.jacobian.cols()!=6 ||
     observation.jacobian.rows()!=observation.residual.size() ||
     !observation.jacobian.allFinite() || !observation.residual.allFinite())
    throw std::invalid_argument("Temporal visual: invalid observation or image/LiDAR time mismatch");
  const uint64_t id=nodes_.back().id;
  if(last_visual_node_==id ||
     std::any_of(visual_edges_.begin(),visual_edges_.end(),[&](const VisualEdge& e){return e.node==id;}))
    throw std::invalid_argument("Temporal visual: image already fused at this node");
  std::set<uint64_t> references;
  for(const auto& reference:observation.references) {
    if(reference.node==id || !references.insert(reference.node).second || !reference.origin.finite() ||
       reference.jacobian.rows()!=observation.residual.size() || reference.jacobian.cols()!=6 ||
       !reference.jacobian.allFinite() ||
       std::none_of(nodes_.begin(),nodes_.end(),[&](const Node& n){return n.id==reference.node;}))
      throw std::invalid_argument("Temporal visual: reference is invalid or outside retained window");
  }
  PoseObservation effective=observation;
  if(config_.visual_degenerate_translation) {
    M3 information=M3::Zero();
    const auto& latest=nodes_.back();
    std::set<size_t> tracked;
    for(const auto& node:nodes_) if(node.id!=id)
      for(const auto& match:node.landmark_matches) tracked.insert(match.landmark);
    for(const auto& match:latest.landmark_matches) if(tracked.count(match.landmark)) {
      const V3& normal=landmarks_[match.landmark].normal;
      information+=match.weight*normal*normal.transpose();
    }
    for(const auto& edge:lidar_edges_) if(edge.source==id) for(const auto& match:edge.matches) {
      const V3 normal=edge.target==map_target_ ? match.plane.normal
          : V3(nodes_[index(edge.target)].state.rotation*match.plane.normal);
      information+=match.weight*normal*normal.transpose();
    }
    effective=weakTranslationObservation(observation,information,config_.visual_degeneracy_ratio);
  }
  Report report;report.status="no_constraints";
  report.visual_rank=effective.jacobian.size() ? effective.jacobian.colPivHouseholderQr().rank() : 0;
  if(effective.jacobian.size()==0 || imageJacobian(effective).squaredNorm()<1e-20) {
    report.marginalized=trim(false);
    marginalCovariance(report);
    nodes_.back().pose_covariance=report.covariance.topLeftCorner<6,6>();
    nodes_.back().covariance_rotation=nodes_.back().state.rotation;
    report.nodes=nodes_.size();report.span=nodes_.back().time-nodes_.front().time;
    last_visual_node_=id;
    return report;
  }
  const auto begin=std::chrono::steady_clock::now();
  std::vector<State> saved;for(const auto& n:nodes_) saved.push_back(n.state);
  const auto saved_landmarks=landmarks_;
  visual_edges_.push_back({id,std::move(effective)});
  try {
    optimize(report);
    report.correction=state().minus(saved.back());
    const V15& d=report.correction;
    if(!d.allFinite() || d.head<3>().norm()>.7853981633974483 || d.segment<3>(3).norm()>2 ||
       d.segment<3>(6).norm()>10 || d.segment<3>(9).norm()>.5 || d.tail<3>().norm()>5) {
      visual_edges_.pop_back();
      for(size_t i=0;i<nodes_.size();++i) nodes_[i].state=saved[i];
      landmarks_=saved_landmarks;
      refresh();report.correction.setZero();report.status="rejected";
    } else report.status="updated";
  } catch(...) {
    if(!visual_edges_.empty() && visual_edges_.back().node==id) visual_edges_.pop_back();
    for(size_t i=0;i<nodes_.size();++i) nodes_[i].state=saved[i];
    landmarks_=saved_landmarks;
    refresh();throw;
  }
  // Consume the reference factor before dropping its non-keyframe pose. The
  // square-root marginal retains its cross-information exactly once. Do this
  // outside the optimization rollback: marginalization changes graph topology.
  report.marginalized=trim(false);
  marginalCovariance(report);
  nodes_.back().pose_covariance=report.covariance.topLeftCorner<6,6>();
  nodes_.back().covariance_rotation=nodes_.back().state.rotation;
  report.nodes=nodes_.size();report.span=nodes_.back().time-nodes_.front().time;
  report.solve_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
  last_visual_node_=id;
  return report;
}
} // namespace temporal_lio

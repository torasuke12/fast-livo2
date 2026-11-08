#include "temporal_lio.h"
#include <Eigen/Cholesky>
#include <Eigen/QR>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace temporal_lio {
int Window::index(uint64_t id) const {
  for(size_t i=0;i<nodes_.size();++i) if(nodes_[i].id==id) return static_cast<int>(i);
  throw std::logic_error("Temporal LIO: dangling factor");
}

void Window::priorFactor(Matrix& jacobian,Vector& residual) const {
  const int pose_dim=15*nodes_.size(),prior_pose_dim=15*prior_origins_.size();
  const int dim=pose_dim+3*landmarks_.size(),pd=prior_jacobian_.cols();
  Vector delta(pd);Matrix prior=prior_jacobian_;
  // The chart transport is block diagonal: only rotation (3 columns) and
  // plane normal (2 columns) differ from identity. Avoid a dense cubic
  // multiplication for hundreds of otherwise unchanged prior columns.
  for(size_t i=0;i<prior_origins_.size();++i) {
    delta.segment<15>(15*i)=nodes_[i].state.minus(prior_origins_[i]);
    prior.middleCols(15*i,3)=prior_jacobian_.middleCols(15*i,3)*rightJacobianInverse(delta.segment<3>(15*i));
  }
  for(size_t i=0;i<prior_landmark_origins_.size();++i) {
    const auto& origin=prior_landmark_origins_[i];const auto& landmark=landmarks_[i];
    delta.segment<2>(prior_pose_dim+3*i)=planeBasis(origin.normal).transpose()*(landmark.normal-origin.normal);
    delta[prior_pose_dim+3*i+2]=landmark.distance-origin.distance;
    prior.middleCols(prior_pose_dim+3*i,2)=prior_jacobian_.middleCols(prior_pose_dim+3*i,2)*
        (planeBasis(origin.normal).transpose()*planeBasis(landmark.normal));
  }
  residual=prior_jacobian_*delta+prior_residual_;
  jacobian=Matrix::Zero(prior.rows(),dim);
  jacobian.leftCols(prior_pose_dim)=prior.leftCols(prior_pose_dim);
  if(pd>prior_pose_dim) jacobian.rightCols(pd-prior_pose_dim)=prior.rightCols(pd-prior_pose_dim);
}

void Window::reanchorPlanes(uint64_t next_anchor) {
  if(plane_anchor_id_==next_anchor) return;
  const int previous=index(plane_anchor_id_),next=index(next_anchor);
  if(!landmarks_.empty()) {
    Matrix j;Vector r;priorFactor(j,r);
    const V3 shift=nodes_[previous].state.position-nodes_[next].state.position;
    const int pose_dim=15*nodes_.size();
    for(size_t i=0;i<landmarks_.size();++i) {
      auto& plane=landmarks_[i];const int col=pose_dim+3*i;
      const Vector distance_column=j.col(col+2);
      // d_new = d_old + n.(p_old - p_new). Transform every prior cross term
      // into the new chart before the old anchor is eliminated.
      j.block(0,15*previous+3,j.rows(),3)-=distance_column*plane.normal.transpose();
      j.block(0,15*next+3,j.rows(),3)+=distance_column*plane.normal.transpose();
      j.middleCols(col,2)-=distance_column*(shift.transpose()*planeBasis(plane.normal));
      plane.distance+=plane.normal.dot(shift);
    }
    prior_jacobian_=std::move(j);prior_residual_=std::move(r);
    prior_origins_.clear();for(const auto& node:nodes_) prior_origins_.push_back(node.state);
    prior_landmark_origins_=landmarks_;
  }
  plane_anchor_id_=next_anchor;
}

void Window::linearize(Matrix& jacobian, Vector& residual, double& cost, int touching) const {
  struct Block { int a,b; Matrix jacobian; Vector residual; bool pose_only; };
  std::vector<Block> blocks;
  const int pose_dim=15*nodes_.size(),dim=pose_dim+3*landmarks_.size();
  Matrix prior;Vector prior_residual;priorFactor(prior,prior_residual);
  cost=0.5*prior_residual.squaredNorm();
  int rows=prior_jacobian_.rows();
  for(const auto& edge:imu_edges_) {
    const int ia=index(edge.a),ib=index(edge.b);
    if(touching>=0 && ia!=touching && ib!=touching) continue;
    const State& a=nodes_[ia].state; const State& z=nodes_[ib].state;
    const V15 r=imuResidual(a,z,edge.delta,config_.gravity);
    const auto j=imuJacobians(a,z,edge.delta,config_.gravity);
    Eigen::LLT<M15> whitening(edge.delta.covariance);
    Block block{ia,ib,whitening.matrixL().solve(j),whitening.matrixL().solve(r),false};
    cost+=0.5*block.residual.squaredNorm();rows+=15;blocks.push_back(std::move(block));
  }
  for(const auto& edge:lidar_edges_) {
    const int ia=index(edge.source),ib=edge.target==map_target_ ? -1 : index(edge.target);
    if(touching>=0 && ia!=touching && ib!=touching) continue;
    if(edge.matches.empty()) continue;
    const auto& source=nodes_[ia];const State fixed_state;
    const auto& a=source.state;const auto& z=ib>=0 ? nodes_[ib].state : fixed_state;
    const int width=config_.consistent_deskew ? 15 : 6;
    const int columns=ib>=0 ? 2*width : width;
    Matrix augmented=Matrix::Zero(edge.matches.size(),columns+1);int row=0;
    for(const auto& match:edge.matches) {
      const V3& p=source.points[match.source];
      const V3 q=z.rotation.transpose()*(a.rotation*p+a.position-z.position);
      const V3 nw=z.rotation*match.plane.normal;
      const double r=match.plane.normal.dot(q-match.plane.center);
      const double normalized=std::abs(r)*std::sqrt(match.weight);
      const double robust=normalized>config_.huber_delta ? config_.huber_delta/normalized : 1.0;
      const double sqrt_weight=std::sqrt(match.weight*robust);
      augmented.block<1,3>(row,0)=-sqrt_weight*nw.transpose()*a.rotation*skew(p);
      augmented.block<1,3>(row,3)=sqrt_weight*nw.transpose();
      if(ib>=0) {
        augmented.block<1,3>(row,width)=sqrt_weight*match.plane.normal.transpose()*skew(q);
        augmented.block<1,3>(row,width+3)=-sqrt_weight*nw.transpose();
      }
      if(config_.consistent_deskew) {
        augmented.block<1,15>(row,0)+=sqrt_weight*nw.transpose()*a.rotation*source.point_jacobians.at(match.source);
        if(ib>=0) augmented.block<1,15>(row,15)+=sqrt_weight*((q-match.plane.center).transpose()*match.plane.normal_jacobian
            -match.plane.normal.transpose()*match.plane.center_jacobian);
      }
      augmented(row++,columns)=sqrt_weight*r;
      cost+=normalized<=config_.huber_delta ? 0.5*normalized*normalized : config_.huber_delta*(normalized-0.5*config_.huber_delta);
    }
    // Compress returns to at most 13 (pose) or 31 (deskew) square-root rows.
    // the window. No A^T A or subtraction of nearly equal information blocks.
    Eigen::HouseholderQR<Matrix> qr(augmented);
    const int count=std::min<int>(augmented.rows(),columns+1);
    const Matrix root=qr.matrixQR().topRows(count).triangularView<Eigen::Upper>();
    blocks.push_back({ia,ib,root.leftCols(columns),root.col(columns),!config_.consistent_deskew});rows+=count;
  }
  for(size_t i=0;i<nodes_.size();++i) {
    if(touching>=0 && static_cast<int>(i)!=touching) continue;
    const auto& node=nodes_[i];
    if(node.structure.empty()) continue;
    Matrix augmented=Matrix::Zero(3*node.structure.size(),16);int row=0;
    for(const auto& observation:node.structure) {
      const auto& plane=observation.plane;const double weight=std::sqrt(observation.weight);
      Eigen::Matrix<double,3,15> j=node.state.rotation*plane.normal_jacobian;
      j.leftCols<3>()-=node.state.rotation*skew(plane.normal);
      const V3 residual=weight*observation.axis.cross(node.state.rotation*plane.normal);
      augmented.block<3,15>(row,0)=weight*skew(observation.axis)*j;
      augmented.block<3,1>(row,15)=residual;row+=3;cost+=.5*residual.squaredNorm();
    }
    Eigen::HouseholderQR<Matrix> qr(augmented);
    const int count=std::min<int>(augmented.rows(),16);
    const Matrix root=qr.matrixQR().topRows(count).triangularView<Eigen::Upper>();
    blocks.push_back({static_cast<int>(i),-1,root.leftCols(15),root.col(15),false});rows+=count;
  }
  struct LandmarkBlock { int node; std::vector<size_t> ids; Matrix root; };
  std::vector<LandmarkBlock> landmark_blocks;
  const int anchor=landmarks_.empty() ? -1 : index(plane_anchor_id_);
  for(size_t i=0;i<nodes_.size();++i) {
    if(touching>=0 && static_cast<int>(i)!=touching && anchor!=touching) continue;
    const auto& node=nodes_[i];
    if(node.landmark_matches.empty()) continue;
    std::map<size_t,int> columns;
    std::vector<size_t> ids;
    for(const auto& match:node.landmark_matches) if(!columns.count(match.landmark)) {
      columns.emplace(match.landmark,15+3*ids.size());ids.push_back(match.landmark);
    }
    const int width=15+3*ids.size();
    Matrix augmented=Matrix::Zero(node.landmark_matches.size(),width+1);int row=0;
    for(const auto& match:node.landmark_matches) {
      const auto& landmark=landmarks_.at(match.landmark);
      const auto& p=node.points.at(match.source);
      const V3& n=landmark.normal;const auto& state=node.state;
      const V3 relative=state.rotation*p+(state.position-nodes_[anchor].state.position);
      const double r=n.dot(relative)-landmark.distance;
      const double normalized=std::abs(r)*std::sqrt(match.weight);
      const double robust=normalized>config_.huber_delta ? config_.huber_delta/normalized : 1.;
      const double weight=std::sqrt(match.weight*robust);
      augmented.block<1,15>(row,0)=weight*n.transpose()*state.rotation*node.point_jacobians.at(match.slot);
      augmented.block<1,3>(row,0)-=weight*n.transpose()*state.rotation*skew(p);
      augmented.block<1,3>(row,3)+=weight*n.transpose();
      augmented.block<1,2>(row,columns.at(match.landmark))=weight*relative.transpose()*planeBasis(n);
      augmented(row,columns.at(match.landmark)+2)=-weight;
      augmented(row++,width)=weight*r;
      cost+=normalized<=config_.huber_delta ? .5*normalized*normalized : config_.huber_delta*(normalized-.5*config_.huber_delta);
    }
    Eigen::HouseholderQR<Matrix> qr(augmented);
    const int count=std::min<int>(augmented.rows(),width+1);
    Matrix root=qr.matrixQR().topRows(count).triangularView<Eigen::Upper>();
    landmark_blocks.push_back({static_cast<int>(i),std::move(ids),std::move(root)});rows+=count;
  }
  const int binary_rows=rows;
  const auto visual_touches=[&](const VisualEdge& edge) {
    return touching<0 || index(edge.node)==touching ||
      std::any_of(edge.observation.references.begin(),edge.observation.references.end(),
                  [&](const PoseObservation::Reference& reference){return index(reference.node)==touching;});
  };
  for(const auto& edge:visual_edges_)
    if(visual_touches(edge)) rows+=edge.observation.residual.size();
  jacobian=Matrix::Zero(rows,dim);residual=Vector::Zero(rows);
  jacobian.topRows(prior.rows())=prior;
  residual.head(prior_residual.size())=prior_residual;
  int row=prior_residual.size();
  for(const auto& block:blocks) {
    const int n=block.residual.size(),width=block.pose_only ? 6 : 15;
    jacobian.block(row,15*block.a,n,width)=block.jacobian.leftCols(width);
    if(block.b>=0) jacobian.block(row,15*block.b,n,width)=block.jacobian.rightCols(width);
    residual.segment(row,n)=block.residual;row+=n;
  }
  for(const auto& block:landmark_blocks) {
    const int n=block.root.rows();
    jacobian.block(row,15*block.node,n,15)=block.root.leftCols(15);
    // The anchor is estimated jointly. Derive its translation columns from
    // the compressed source columns to preserve their exact cancellation.
    jacobian.block(row,15*anchor+3,n,3)-=block.root.middleCols(3,3);
    for(size_t k=0;k<block.ids.size();++k) jacobian.block(row,pose_dim+3*block.ids[k],n,3)=block.root.middleCols(15+3*k,3);
    residual.segment(row,n)=block.root.col(block.root.cols()-1);row+=n;
  }
  if(row!=binary_rows) throw std::logic_error("Temporal LIO: factor row mismatch");
  for(const auto& edge:visual_edges_) {
    const int i=index(edge.node);
    if(!visual_touches(edge)) continue;
    const auto& o=edge.observation;
    const V15 delta=nodes_[i].state.minus(o.origin);
    Eigen::Matrix<double,6,6> transport=Eigen::Matrix<double,6,6>::Identity();
    transport.topLeftCorner<3,3>()=rightJacobianInverse(delta.head<3>());
    Vector r=o.jacobian*delta.head<6>()+o.residual;
    jacobian.block(row,15*i,r.size(),6)=o.jacobian*transport;
    for(const auto& reference:o.references) {
      const int reference_index=index(reference.node);
      const V15 reference_delta=nodes_[reference_index].state.minus(reference.origin);
      M6 reference_transport=M6::Identity();
      reference_transport.topLeftCorner<3,3>()=rightJacobianInverse(reference_delta.head<3>());
      jacobian.block(row,15*reference_index,r.size(),6)=reference.jacobian*reference_transport;
      r+=reference.jacobian*reference_delta.head<6>();
    }
    residual.segment(row,r.size())=r;cost+=0.5*r.squaredNorm();row+=r.size();
  }
  if(!jacobian.allFinite() || !residual.allFinite() || !std::isfinite(cost))
    throw std::runtime_error("Temporal LIO: nonfinite square-root system");
}

double Window::trialCost() const {
  const int pd=prior_jacobian_.cols();Vector delta(pd);
  for(size_t i=0;i<prior_origins_.size();++i) delta.segment<15>(15*i)=nodes_[i].state.minus(prior_origins_[i]);
  for(size_t i=0;i<prior_landmark_origins_.size();++i) {
    const int col=15*prior_origins_.size()+3*i;
    delta.segment<2>(col)=planeBasis(prior_landmark_origins_[i].normal).transpose()*(landmarks_[i].normal-prior_landmark_origins_[i].normal);
    delta[col+2]=landmarks_[i].distance-prior_landmark_origins_[i].distance;
  }
  double cost=.5*(prior_jacobian_*delta+prior_residual_).squaredNorm();
  for(const auto& edge:imu_edges_) {
    const auto residual=imuResidual(nodes_[index(edge.a)].state,nodes_[index(edge.b)].state,edge.delta,config_.gravity);
    cost+=.5*residual.dot(edge.delta.information*residual);
  }
  std::vector<std::vector<V3>> points;
  for(const auto& node:nodes_) points.push_back(deskew(node.raw,node.scan_imu,node.time,node.state,config_));
  std::map<std::pair<uint64_t,VoxelKey>,Plane> refitted;
  for(const auto& edge:lidar_edges_) {
    const int ia=index(edge.source),ib=edge.target==map_target_ ? -1 : index(edge.target);
    const State fixed_state;const auto& a=nodes_[ia].state;const auto& b=ib>=0 ? nodes_[ib].state : fixed_state;
    for(const auto& match:edge.matches) {
      const auto key=std::make_pair(edge.target,match.key);
      auto it=refitted.find(key);
      if(it==refitted.end()) it=refitted.emplace(key,ib<0 ? match.plane : refitPlane(nodes_[ib].initialization_reference.empty()
          ? points[ib] : nodes_[ib].initialization_reference,match.plane)).first;
      const auto& plane=it->second;
      const V3 q=b.rotation.transpose()*(a.rotation*points[ia][match.source]+a.position-b.position);
      const double residual=std::abs(plane.normal.dot(q-plane.center))*std::sqrt(match.weight);
      cost+=residual<=config_.huber_delta ? .5*residual*residual : config_.huber_delta*(residual-.5*config_.huber_delta);
    }
  }
  for(const auto& edge:visual_edges_) {
    const auto& o=edge.observation;
    Vector r=o.jacobian*nodes_[index(edge.node)].state.minus(o.origin).head<6>()+o.residual;
    for(const auto& reference:o.references)
      r+=reference.jacobian*nodes_[index(reference.node)].state.minus(reference.origin).head<6>();
    cost+=.5*r.squaredNorm();
  }
  for(size_t i=0;i<nodes_.size();++i) {
    const auto& node=nodes_[i];
    for(const auto& match:node.landmark_matches) {
      const auto& landmark=landmarks_.at(match.landmark);
      const double r=std::abs(landmark.normal.dot(node.state.rotation*points[i].at(match.source)+
          (node.state.position-nodes_.at(index(plane_anchor_id_)).state.position))
          -landmark.distance)*std::sqrt(match.weight);
      cost+=r<=config_.huber_delta ? .5*r*r : config_.huber_delta*(r-.5*config_.huber_delta);
    }
    for(const auto& observation:node.structure) {
      const auto key=std::make_pair(node.id,observation.key);
      auto it=refitted.find(key);
      if(it==refitted.end()) it=refitted.emplace(key,refitPlane(node.initialization_reference.empty()
          ? points[i] : node.initialization_reference,observation.plane)).first;
      cost+=.5*observation.weight*observation.axis.cross(node.state.rotation*it->second.normal).squaredNorm();
    }
  }
  if(!std::isfinite(cost)) throw std::runtime_error("Nonfinite temporal trial cost");
  return cost;
}

bool schurComplement(const Matrix& h,const Vector& b,int offset,int count,Matrix& out,Vector& rhs) {
  const int n=h.rows();
  if(h.cols()!=n || b.size()!=n || offset<0 || count<=0 || offset+count>n ||
     !h.allFinite() || !b.allFinite()) return false;
  std::vector<int> keep;
  for(int i=0;i<n;++i) if(i<offset || i>=offset+count) keep.push_back(i);
  Matrix cross(keep.size(),count); Vector remaining(keep.size()); out.resize(keep.size(),keep.size());
  for(size_t i=0;i<keep.size();++i) {
    cross.row(i)=h.block(keep[i],offset,1,count); remaining[i]=b[keep[i]];
    for(size_t j=0;j<keep.size();++j) out(i,j)=h(keep[i],keep[j]);
  }
  const Matrix eliminated=h.block(offset,offset,count,count);
  Eigen::LLT<Matrix> solve(eliminated);
  if(solve.info()!=Eigen::Success) return false;
  out-=cross*solve.solve(cross.transpose()); rhs=remaining-cross*solve.solve(b.segment(offset,count));
  out=0.5*(out+out.transpose()).eval();
  return out.allFinite() && rhs.allFinite();
}

void Window::marginalize(int dropped) {
  if(config_.structural_landmarks && nodes_.at(dropped).id==plane_anchor_id_) {
    if(nodes_.size()<2) throw std::logic_error("Cannot eliminate the last plane anchor");
    reanchorPlanes(nodes_[dropped==0 ? 1 : 0].id);
  }
  Matrix jacobian;Vector residual;double cost;
  // Only factors touching the removed node enter the prior. Retained factors
  // stay active and are never also counted in the prior.
  linearize(jacobian,residual,cost,dropped);
  const int dim=jacobian.cols(),remaining=dim-15;
  Matrix reordered(jacobian.rows(),dim+1);
  reordered.leftCols(15)=jacobian.middleCols(15*dropped,15);
  int col=15;
  for(int i=0;i<static_cast<int>(nodes_.size());++i) if(i!=dropped) {
    reordered.middleCols(col,15)=jacobian.middleCols(15*i,15);col+=15;
  }
  if(!landmarks_.empty()) reordered.middleCols(col,3*landmarks_.size())=jacobian.rightCols(3*landmarks_.size());
  reordered.col(dim)=residual;
  Eigen::HouseholderQR<Matrix> qr(reordered);
  const int root_rows=std::min<int>(reordered.rows(),dim+1);
  if(root_rows<15 || (qr.matrixQR().diagonal().head(15).array().abs()<1e-15).any())
    throw std::runtime_error("Temporal LIO: singular eliminated state");
  const Matrix root=qr.matrixQR().topRows(root_rows).triangularView<Eigen::Upper>();
  prior_jacobian_=root.block(15,15,root_rows-15,remaining);
  prior_residual_=root.block(15,dim,root_rows-15,1);
  const auto id=nodes_.at(dropped).id;
  // Only finalized keyframes enter the map. No active source scan is matched
  // against a map made from its own returns. The first bootstrap was inserted
  // once during initialization and must not be inserted a second time.
  if(config_.persistent_map && nodes_[dropped].keyframe && id!=0) insertMap(nodes_[dropped]);
  imu_edges_.erase(std::remove_if(imu_edges_.begin(),imu_edges_.end(),[&](const ImuEdge& e){return e.a==id || e.b==id;}),imu_edges_.end());
  lidar_edges_.erase(std::remove_if(lidar_edges_.begin(),lidar_edges_.end(),[&](const LidarEdge& e){return e.source==id || e.target==id;}),lidar_edges_.end());
  visual_edges_.erase(std::remove_if(visual_edges_.begin(),visual_edges_.end(),[&](const VisualEdge& e){
    return e.node==id || std::any_of(e.observation.references.begin(),e.observation.references.end(),
                                  [&](const PoseObservation::Reference& r){return r.node==id;});
  }),visual_edges_.end());
  nodes_.erase(nodes_.begin()+dropped);
  prior_landmark_origins_.clear();
  for(const auto& landmark:landmarks_) prior_landmark_origins_.push_back(landmark);
  prior_origins_.clear();
  for(const auto& n:nodes_) prior_origins_.push_back(n.state);
  // Retain a structural plane only while it is observed in the active
  // window. Eliminate its distance with its correlations, so the estimator
  // cannot recognize an old room or close a loop when that room is revisited.
  if(config_.structural_landmarks) {
    std::vector<bool> observed(landmarks_.size(),false);
    for(const auto& node:nodes_) for(const auto& match:node.landmark_matches) observed.at(match.landmark)=true;
    std::vector<size_t> removed,kept;
    for(size_t i=0;i<observed.size();++i) (observed[i] ? kept : removed).push_back(i);
    if(!removed.empty()) {
      const int pose_columns=15*nodes_.size(),count=3*removed.size(),width=prior_jacobian_.cols();
      Matrix system(prior_jacobian_.rows(),width+1);
      for(size_t i=0;i<removed.size();++i) system.middleCols(3*i,3)=prior_jacobian_.middleCols(pose_columns+3*removed[i],3);
      system.middleCols(count,pose_columns)=prior_jacobian_.leftCols(pose_columns);
      for(size_t i=0;i<kept.size();++i) system.middleCols(count+pose_columns+3*i,3)=prior_jacobian_.middleCols(pose_columns+3*kept[i],3);
      system.col(width)=prior_residual_;
      Eigen::HouseholderQR<Matrix> eliminate(system);
      const int height=std::min<int>(system.rows(),width+1);
      if(height<count || (eliminate.matrixQR().diagonal().head(count).array().abs()<1e-15).any())
        throw std::runtime_error("Temporal LIO: singular retired plane");
      const Matrix root=eliminate.matrixQR().topRows(height).triangularView<Eigen::Upper>();
      prior_jacobian_=root.block(count,count,height-count,width-count);
      prior_residual_=root.block(count,width,height-count,1);
      std::vector<Landmark> retained;
      std::vector<size_t> remap(landmarks_.size());
      prior_landmark_origins_.clear();
      for(size_t i:kept) {
        remap[i]=retained.size();retained.push_back(landmarks_[i]);
        prior_landmark_origins_.push_back(landmarks_[i]);
      }
      landmarks_=std::move(retained);
      for(auto& node:nodes_) for(auto& match:node.landmark_matches) match.landmark=remap.at(match.landmark);
    }
  }
}
} // namespace temporal_lio

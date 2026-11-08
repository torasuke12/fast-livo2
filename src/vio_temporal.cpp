#include "vio.h"

void VIOManager::captureTemporalObservation(const MatrixXd& j,const VectorXd& r) {
  temporal_lio::State origin;
  origin.rotation=state->rot_end;origin.position=state->pos_end;
  origin.velocity=state->vel_end;origin.gyro_bias=state->bias_g;origin.accel_bias=state->bias_a;
  MatrixXd current=j;VectorXd residual=r;
  std::vector<temporal_lio::PoseObservation::Reference> references;
  if(temporal_joint_reference) {
    std::map<uint64_t,size_t> slots;
    for(size_t i=0;i<visual_submap->voxel_points.size();++i) {
      const auto* point=visual_submap->voxel_points[i];
      const int row=i*patch_size_total;
      if(row+patch_size_total>r.size()) throw std::logic_error("Visual patch rows do not match point anchors");
      const auto anchor=point ? temporal_anchor_states.find(point->temporal_anchor_) : temporal_anchor_states.end();
      if(anchor==temporal_anchor_states.end() || anchor->first==temporal_node_id) {
        current.middleRows(row,patch_size_total).setZero();residual.segment(row,patch_size_total).setZero();continue;
      }
      if(current.middleRows(row,patch_size_total).squaredNorm()==0) continue;
      auto inserted=slots.emplace(anchor->first,references.size());
      if(inserted.second) references.push_back({anchor->first,anchor->second,MatrixXd::Zero(r.size(),6)});
      references[inserted.first->second].jacobian.middleRows(row,patch_size_total)=
        temporal_lio::pointAnchorJacobian(anchor->second,point->pos_,current.block(row,3,patch_size_total,3));
    }
  }
  temporal_observation=temporal_lio::photometricObservation(origin,current,residual,img_point_cov,references);
  temporal_pixels=r.size();
}

void VIOManager::synchronizeTemporalMap(const std::vector<temporal_lio::Node>& nodes) {
  std::map<uint64_t,SE3> changes;
  std::map<uint64_t,temporal_lio::State> next;
  for(const auto& node:nodes) {
    const auto previous=temporal_anchor_states.find(node.id);
    if(previous!=temporal_anchor_states.end()) {
      const M3D rotation=node.state.rotation*previous->second.rotation.transpose();
      changes.emplace(node.id,SE3(rotation,node.state.position-rotation*previous->second.position));
    }
    next.emplace(node.id,node.state);
  }
  // A marginalized anchor is frozen at its last published estimate, just as
  // the trajectory history is. Retained anchors follow the smoother.
  temporal_anchor_states=std::move(next);
  if(!nodes.empty()) temporal_node_id=nodes.back().id;
  if(changes.empty() && !temporal_joint_reference) return;
  std::vector<VisualPoint*> points;
  for(auto& entry:feat_map) {
    for(auto* point:entry.second->voxel_points) {
      // Relative image factors only use retained reference poses. Once the
      // pose leaves the window its factors are in the marginal prior; create
      // new measured references instead of treating an old point as exact.
      if(temporal_joint_reference && !temporal_anchor_states.count(point->temporal_anchor_)) {
        delete point;continue;
      }
      const auto change=changes.find(point->temporal_anchor_);
      if(change!=changes.end()) {
        const auto& transform=change->second;
        const M3D rotation=transform.rotationMatrix();
        point->pos_=transform*point->pos_;
        point->normal_=rotation*point->normal_;
        point->previous_normal_=rotation*point->previous_normal_;
        point->covariance_=rotation*point->covariance_*rotation.transpose();
      }
      for(auto* feature:point->obs_) {
        const auto frame_change=changes.find(feature->temporal_anchor_);
        if(frame_change!=changes.end()) feature->T_f_w_=feature->T_f_w_*frame_change->second.inverse();
      }
      points.push_back(point);
    }
    // Transfer ownership before destroying the old spatial buckets.
    entry.second->voxel_points.clear();delete entry.second;
  }
  feat_map.clear();sub_feat_map.clear();
  for(auto* point:points) insertPointIntoVoxelMap(point);
  for(auto& entry:warp_map) delete entry.second;
  warp_map.clear();has_ref_patch_cache=false;
}

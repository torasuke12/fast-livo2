#include "LIVMapper.h"
#include <filesystem>
#include <iomanip>

void LIVMapper::prepareTemporalVisualPoints() {
  _pv_list.clear();pcl_w_wait_pub->clear();
  const auto& node=temporal_window_->nodes().back();
  const auto& s=node.state;
  // Supply actual deskewed returns and their scan-plane normals to the same
  // visual frontend. No legacy IEKF update or stale bootstrap octree is used.
  for(size_t i=0;i<node.points.size();++i) {
    const V3D world=s.rotation*node.points[i]+s.position;
    PointType p{};p.x=world.x();p.y=world.y();p.z=world.z();p.intensity=node.raw[i].intensity;
    pcl_w_wait_pub->push_back(p);
  }
  // The visual frontend chooses a Shi-Tomasi feature per image grid cell.
  // A geometric solver's six returns per plane are not a texture selection:
  // they can omit a useful image corner even when its measured depth exists.
  // Keep the two budgets separate; this does not add any LiDAR factors.
  _pv_list.reserve(temporal_visual_full_points_ ? node.points.size() : node.source_indices.size());
  const auto append_visual_point=[&](size_t i) {
    const auto plane=node.planes.find(temporal_lio::voxelKey(node.points[i],temporal_config_.voxel_size));
    if(plane==node.planes.end()) return;
    pointWithVar p;p.point_i=node.points[i];p.point_b=extR.transpose()*(p.point_i-extT);
    p.point_w=s.rotation*p.point_i+s.position;p.normal=s.rotation*plane->second.normal;
    p.var_nostate=M3D::Identity()*(temporal_config_.point_sigma*temporal_config_.point_sigma+plane->second.variance);
    Eigen::Matrix<double,3,6> j;j<<-s.rotation*temporal_lio::skew(p.point_i),M3D::Identity();
    p.var=p.var_nostate+j*_state.cov.topLeftCorner<6,6>()*j.transpose();
    _pv_list.push_back(p);
  };
  if(temporal_visual_full_points_)
    for(size_t i=0;i<node.points.size();++i) append_visual_point(i);
  else
    for(size_t i:node.source_indices) append_visual_point(i);
}

void LIVMapper::handleTemporalVIO() {
  if(!temporal_window_ || !lidar_map_inited) return;
  const double time=LidarMeasures.measures.back().vio_time;
  if(std::abs(time-temporal_window_->nodes().back().time)>1e-6)
    throw std::runtime_error("Temporal LIVO: image does not match latest LiDAR cut time");
  if(!temporal_visual_log_.is_open()) {
    const auto directory=debug_log_dir.empty() ? std::filesystem::path(ROOT_DIR)/"Log" : std::filesystem::path(debug_log_dir);
    temporal_visual_log_.open(directory/"temporal_visual.csv");
    if(!temporal_visual_log_) throw std::runtime_error("Cannot open temporal visual diagnostics");
    temporal_visual_log_<<"time,status,pixels,rows,solve_ms,px,py,pz,vx,vy,vz,bgx,bgy,bgz,bax,bay,baz,inv_exposure";
    for(int i=0;i<15;++i) temporal_visual_log_<<",correction_"<<i;
    for(int i=0;i<15;++i) for(int j=0;j<15;++j) temporal_visual_log_<<",cov_"<<i<<'_'<<j;
    temporal_visual_log_<<",visual_rank,candidate_points,invalid_patch_evaluations,reference_poses,tracked_features,reprojection_inliers,tracking_dt\n"<<std::setprecision(12);
  }
  vio_manager->temporal_joint_reference=temporal_visual_joint_reference_;
  vio_manager->synchronizeTemporalMap(temporal_window_->nodes());
  prepareTemporalVisualPoints();
  const size_t candidate_points=_pv_list.size();
  const StatesGroup before=_state;
  state_propagat=before;
  temporal_lio::Report report;report.status="frontend_rejected";
  const auto covariance=[&]() {
    constexpr int ix[15]={0,1,2,3,4,5,7,8,9,10,11,12,13,14,15};
    for(int i=0;i<15;++i) for(int j=0;j<15;++j) report.covariance(i,j)=_state.cov(ix[i],ix[j]);
  };
  covariance();
  temporal_lio::ReprojectionTracker::Statistics tracking;
  if(temporal_visual_reprojection_) {
    if(!temporal_reprojection_) {
      M3D intrinsics=M3D::Identity();
      intrinsics(0,0)=vio_manager->cam->fx();intrinsics(1,1)=vio_manager->cam->fy();
      intrinsics(0,2)=vio_manager->cam->cx();intrinsics(1,2)=vio_manager->cam->cy();
      temporal_reprojection_=std::make_unique<temporal_lio::ReprojectionTracker>(intrinsics,vio_manager->Rci,vio_manager->Pci);
    }
    const cv::Mat& original=LidarMeasures.measures.back().img;
    if(original.cols!=vio_manager->cam->width() || original.rows!=vio_manager->cam->height())
      throw std::invalid_argument("Reprojection image dimensions do not match calibrated intrinsics");
    cv::Mat gray;
    if(original.channels()==3) cv::cvtColor(original,gray,cv::COLOR_BGR2GRAY);
    else gray=original;
    // The ordinary VIO path constructs this in processFrame. The publisher
    // and updateFrameState require the same calibrated frame in tracked mode.
    vio_manager->new_frame_.reset(new Frame(vio_manager->cam,gray));
    auto observation=temporal_reprojection_->observe(gray,temporal_window_->nodes());
    tracking=temporal_reprojection_->statistics();
    vio_manager->temporal_observation=observation;
    vio_manager->temporal_pixels=observation.residual.size() ? 2*tracking.inliers : 0;
    vio_manager->invalid_patch_evaluations=0;
    if(!temporal_visual_factor_enabled_) {
      observation.jacobian.resize(0,6);
      observation.residual.resize(0);
      observation.references.clear();
    }
    report=temporal_window_->addVisual(time,observation);
    if(!temporal_visual_factor_enabled_) report.status="factor_disabled";
    applyTemporalState(report);prepareTemporalVisualPoints();
    temporal_reprojection_->seed(gray,temporal_window_->nodes().back());
    vio_manager->img_rgb=original.clone();vio_manager->img_cp=original.clone();
    vio_manager->updateFrameState(_state);
  } else {
    // processFrame calls this before adding reference patches. Discard the
    // frontend state posterior: only its image likelihood enters the window.
    vio_manager->temporal_commit=[&]() {
      const double exposure=_state.inv_expo_time;
      const double exposure_variance=_state.cov(6,6);
      _state=before;
      if(vio_manager->temporal_pixels==0) {report.status="no_constraints";return true;}
      auto observation=vio_manager->temporal_observation;
      if(!temporal_visual_factor_enabled_) {
        observation.jacobian.resize(0,6);
        observation.residual.resize(0);
        observation.references.clear();
      }
      report=temporal_window_->addVisual(time,observation);
      if(!temporal_visual_factor_enabled_) report.status="factor_disabled";
      applyTemporalState(report);
      if(report.status=="rejected") return false;
      _state.inv_expo_time=exposure;_state.cov(6,6)=exposure_variance;
      vio_manager->synchronizeTemporalMap(temporal_window_->nodes());
      prepareTemporalVisualPoints();
      return true;
    };
    const bool accepted=vio_manager->processFrame(LidarMeasures.measures.back().img,_pv_list,
        voxelmap_manager->voxel_map_,time-_first_lidar_time);
    vio_manager->temporal_commit={};
    if(!accepted) { _state=before;prepareTemporalVisualPoints(); }
  }
  state_propagat=_state;
  const auto& s=temporal_window_->state();
  temporal_visual_log_<<time-_first_lidar_time<<','<<report.status<<','<<vio_manager->temporal_pixels
    <<','<<vio_manager->temporal_observation.residual.size()<<','<<report.solve_ms;
  for(const auto& v:{s.position,s.velocity,s.gyro_bias,s.accel_bias})
    for(int k=0;k<3;++k) temporal_visual_log_<<','<<v[k];
  temporal_visual_log_<<','<<_state.inv_expo_time;
  for(int i=0;i<15;++i) temporal_visual_log_<<','<<report.correction[i];
  for(int i=0;i<15;++i) for(int j=0;j<15;++j) temporal_visual_log_<<','<<report.covariance(i,j);
  temporal_visual_log_<<','<<report.visual_rank<<','<<candidate_points<<','<<vio_manager->invalid_patch_evaluations
    <<','<<vio_manager->temporal_observation.references.size()<<','<<tracking.tracked<<','<<tracking.inliers<<','<<tracking.dt;
  temporal_visual_log_<<std::endl;
  if(!temporal_visual_log_) throw std::runtime_error("Cannot write temporal visual diagnostics");
  // Keep the retained-state history consistent with the post-image solution.
  for(const auto& n:temporal_window_->nodes()) {
    Eigen::Quaterniond q(n.state.rotation);
    temporal_states_log_<<time-_first_lidar_time<<','<<n.id<<','<<n.time-_first_lidar_time<<','<<n.keyframe
      <<','<<q.w()<<','<<q.x()<<','<<q.y()<<','<<q.z();
    for(const auto& v:{n.state.position,n.state.velocity,n.state.gyro_bias,n.state.accel_bias})
      for(int k=0;k<3;++k) temporal_states_log_<<','<<v[k];
    temporal_states_log_<<'\n';
  }
  temporal_states_log_.flush();
  if(!temporal_states_log_) throw std::runtime_error("Cannot write temporal visual state history");
  if(imu_prop_enable) {
    ekf_finish_once=true;latest_ekf_state=_state;latest_ekf_time=time;state_update_flg=true;
  }
  euler_cur=RotMtoEuler(_state.rot_end);
  geoQuat=tf::createQuaternionMsgFromRollPitchYaw(euler_cur(0),euler_cur(1),euler_cur(2));
  publish_odometry(pubOdomAftMapped);publishTemporalPath(time);
  update_and_publish_map(pcl_w_wait_pub);
  publish_frame_world(pubLaserCloudFullRes,vio_manager);publish_img_rgb(pubImage,vio_manager);
  if(pose_output_en) {
    std::ofstream evo(std::string(ROOT_DIR)+"Log/result/"+seq_name+"_temporal.txt",
        temporal_window_->nodes().back().id ? std::ios::app : std::ios::out);
    if(!evo) throw std::runtime_error("Cannot open temporal visual trajectory output");
    Eigen::Quaterniond q(s.rotation);
    evo<<std::fixed<<std::setprecision(9)<<time<<' '<<s.position.transpose()<<' '
       <<q.x()<<' '<<q.y()<<' '<<q.z()<<' '<<q.w()<<'\n';
  }
  fout_out<<std::setprecision(12)<<time-_first_lidar_time<<' '<<euler_cur.transpose()*57.3<<' '
    <<s.position.transpose()<<' '<<s.velocity.transpose()<<' '<<s.gyro_bias.transpose()<<' '
    <<s.accel_bias.transpose()<<' '<<V3D(_state.inv_expo_time,0,0).transpose()<<' '<<feats_undistort->size()<<std::endl;
}

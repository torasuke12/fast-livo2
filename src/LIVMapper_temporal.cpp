#include "LIVMapper.h"
#include <filesystem>
#include <iomanip>

namespace {
constexpr int state_index[15]={0,1,2,3,4,5,7,8,9,10,11,12,13,14,15};
temporal_lio::State temporalState(const StatesGroup& s) {
  temporal_lio::State t;
  t.rotation=s.rot_end; t.position=s.pos_end; t.velocity=s.vel_end;
  t.gyro_bias=s.bias_g; t.accel_bias=s.bias_a; return t;
}
void writeState(std::ostream& out,const temporal_lio::State& s) {
  Eigen::Quaterniond q(s.rotation);
  out<<','<<q.w()<<','<<q.x()<<','<<q.y()<<','<<q.z();
  for(const auto& v:{s.position,s.velocity,s.gyro_bias,s.accel_bias})
    for(int k=0;k<3;++k) out<<','<<v[k];
}
}

void LIVMapper::configureTemporalLIO() {
  temporal_enabled_=fast_livo2::declareOrGet<bool>(node_,"lio/temporal/enabled",false);
  temporal_initialization_mode_=fast_livo2::declareOrGet<std::string>(node_,"lio/temporal/initialization_mode","single_scan");
  if(temporal_initialization_mode_!="single_scan" && temporal_initialization_mode_!="legacy_voxel")
    throw std::invalid_argument("lio.temporal.initialization_mode must be single_scan or legacy_voxel");
  auto& c=temporal_config_;
  c.window_sec=fast_livo2::declareOrGet<double>(node_,"lio/temporal/window_sec",c.window_sec);
  c.keyframe_sec=fast_livo2::declareOrGet<double>(node_,"lio/temporal/keyframe_sec",c.keyframe_sec);
  c.max_nodes=fast_livo2::declareOrGet<int>(node_,"lio/temporal/max_nodes",c.max_nodes);
  c.iterations=fast_livo2::declareOrGet<int>(node_,"lio/temporal/iterations",c.iterations);
  c.max_source_points=fast_livo2::declareOrGet<int>(node_,"lio/temporal/max_source_points",c.max_source_points);
  c.voxel_size=fast_livo2::declareOrGet<double>(node_,"lio/temporal/voxel_size",c.voxel_size);
  c.sample_size=fast_livo2::declareOrGet<double>(node_,"lio/temporal/sample_size",c.sample_size);
  c.min_plane_points=fast_livo2::declareOrGet<int>(node_,"lio/temporal/min_plane_points",c.min_plane_points);
  c.plane_ratio=fast_livo2::declareOrGet<double>(node_,"lio/temporal/plane_ratio",c.plane_ratio);
  c.min_plane_spread=fast_livo2::declareOrGet<double>(node_,"lio/temporal/min_plane_spread",c.min_plane_spread);
  c.max_residual=fast_livo2::declareOrGet<double>(node_,"lio/temporal/max_residual",c.max_residual);
  c.point_sigma=fast_livo2::declareOrGet<double>(node_,"lio/temporal/point_sigma",c.point_sigma);
  c.huber_delta=fast_livo2::declareOrGet<double>(node_,"lio/temporal/huber_delta",c.huber_delta);
  c.max_imu_gap=fast_livo2::declareOrGet<double>(node_,"lio/temporal/max_imu_gap",c.max_imu_gap);
  c.accel_saturation=fast_livo2::declareOrGet<double>(node_,"lio/temporal/accel_saturation",0.);
  c.accel_saturation_variance=fast_livo2::declareOrGet<double>(node_,"lio/temporal/accel_saturation_variance",2500.);
  c.visual_degenerate_translation=fast_livo2::declareOrGet<bool>(node_,"lio/temporal/visual_degenerate_translation",false);
  c.visual_degeneracy_ratio=fast_livo2::declareOrGet<double>(node_,"lio/temporal/visual_degeneracy_ratio",.05);
  c.consistent_deskew=fast_livo2::declareOrGet<bool>(node_,"lio/temporal/consistent_deskew",false);
  c.plane_uncertainty=fast_livo2::declareOrGet<bool>(node_,"lio/temporal/plane_uncertainty",false);
  c.persistent_map=fast_livo2::declareOrGet<bool>(node_,"lio/temporal/persistent_map",false);
  c.map_max_planes=fast_livo2::declareOrGet<int>(node_,"lio/temporal/map_max_planes",120000);
  c.structural_landmarks=fast_livo2::declareOrGet<bool>(node_,"lio/temporal/structural_landmarks",false);
  c.max_landmarks=fast_livo2::declareOrGet<int>(node_,"lio/temporal/max_landmarks",256);
  c.structure_enabled=fast_livo2::declareOrGet<bool>(node_,"lio/temporal/structure/enabled",false);
  c.structure_sigma=fast_livo2::declareOrGet<double>(node_,"lio/temporal/structure/sigma_deg",0.5)*M_PI/180.;
  c.structure_max_angle=fast_livo2::declareOrGet<double>(node_,"lio/temporal/structure/max_angle_deg",8.)*M_PI/180.;
  c.structure_min_planes=fast_livo2::declareOrGet<int>(node_,"lio/temporal/structure/min_planes",8);
  c.gyro_variance=V3D::Constant(gyr_cov); c.accel_variance=V3D::Constant(acc_cov);
  const bool visual=fast_livo2::declareOrGet<bool>(node_,"lio/temporal/visual_fusion",img_en!=0);
  temporal_image_interval_sec_=fast_livo2::declareOrGet<double>(node_,"lio/temporal/image_interval_sec",0.1);
  temporal_visual_full_points_=fast_livo2::declareOrGet<bool>(node_,"lio/temporal/visual_full_points",false);
  temporal_visual_joint_reference_=fast_livo2::declareOrGet<bool>(node_,"lio/temporal/visual_joint_reference",false);
  temporal_visual_reprojection_=fast_livo2::declareOrGet<bool>(node_,"lio/temporal/visual_reprojection",false);
  temporal_visual_factor_enabled_=fast_livo2::declareOrGet<bool>(node_,"lio/temporal/visual_factor_enabled",true);
  if(!temporal_enabled_) return;
  if(temporal_visual_reprojection_) {
    if(!img_en || !visual)
      throw std::invalid_argument("Visual reprojection requires images and visual fusion");
    if(!std::isfinite(temporal_image_interval_sec_) || temporal_image_interval_sec_>.3)
      throw std::invalid_argument("Visual reprojection requires image_interval_sec <= 0.3 for consecutive tracking");
    for(const auto* coefficient:{"cam_d0","cam_d1","cam_d2","cam_d3","cam_d4"})
      if(std::abs(fast_livo2::declareOrGet<double>(node_,std::string("camera/")+coefficient,0.))>1e-12)
        throw std::invalid_argument("Visual reprojection requires rectified pinhole images (zero distortion)");
  }
  if(temporal_visual_joint_reference_ && inverse_composition_en)
    throw std::invalid_argument("Joint visual references require vio.inverse_composition_en=false");
  if(!std::isfinite(temporal_image_interval_sec_) || temporal_image_interval_sec_<0.02 || temporal_image_interval_sec_>1.0)
    throw std::invalid_argument("lio.temporal.image_interval_sec must be in [0.02, 1.0]");
  if(img_en && !visual) throw std::runtime_error("Temporal LIVO requires lio.temporal.visual_fusion=true");
  if(!imu_en || !lidar_en || gravity_est_en || !ba_bg_est_en)
    throw std::runtime_error("Temporal window requires IMU/LiDAR/bias estimation enabled, and fixed gravity (imu.gravity_est_en=false)");
  c.validate();
  RCLCPP_INFO(node_->get_logger(),"Temporal LIO enabled: %.2f s window, %.2f s keyframes; joint pose/velocity/gyro-bias/accel-bias. Legacy IEKF and X-ICP updates bypassed.",c.window_sec,c.keyframe_sec);
}

void LIVMapper::collectTemporalImu() {
  auto append=[&](const sensor_msgs::Imu::ConstSharedPtr& p) {
    temporal_lio::ImuSample q;
    q.time=fast_livo2::stampToSec(p->header.stamp);
    if(!temporal_imu_.empty() && q.time<=temporal_imu_.back().time) return;
    q.gyro=V3D(p->angular_velocity.x,p->angular_velocity.y,p->angular_velocity.z);
    q.accel=V3D(p->linear_acceleration.x,p->linear_acceleration.y,p->linear_acceleration.z);
    temporal_imu_.push_back(q);
  };
  for(const auto& p:LidarMeasures.measures.back().imu) append(p);
  // sync_packages leaves the first future IMU in the queue. It is read here
  // without consuming it, so scan-end interpolation has both boundaries.
  if(!imu_buffer.empty()) append(imu_buffer.front());
  if(p_imu->imu_need_init && !temporal_imu_.empty()) {
    const double cutoff=temporal_imu_.back().time-1.0;
    temporal_imu_.erase(temporal_imu_.begin(),std::lower_bound(temporal_imu_.begin(),temporal_imu_.end(),cutoff,
      [](const temporal_lio::ImuSample& q,double t){return q.time<t;}));
  }
}

void LIVMapper::handleTemporalLIO() {
  const double time=LidarMeasures.last_lio_update_time;
  const bool initializing=!temporal_window_;
  std::vector<temporal_lio::V3> initialization_reference;
  const bool legacy_initialization=temporal_initialization_mode_=="legacy_voxel";
  auto trimImu=[&]() {
    const double cutoff=time-0.2;
    temporal_imu_.erase(temporal_imu_.begin(),std::lower_bound(temporal_imu_.begin(),temporal_imu_.end(),cutoff,
      [](const temporal_lio::ImuSample& q,double t){return q.time<t;}));
  };
  auto logInitialization=[&](bool ready,size_t reference_points,size_t reference_planes) {
    const auto& s=voxelmap_manager->initializationStatus();
    temporal_initialization_log_<<time-_first_lidar_time<<','<<temporal_initialization_mode_
      <<','<<(ready ? "ready" : "waiting")<<','<<legacy_initialization
      <<','<<s.scans<<','<<s.points<<','<<s.planes;
    for(int k=0;k<3;++k) temporal_initialization_log_<<','<<s.normal_eigenvalues[k];
    temporal_initialization_log_<<','<<s.duration_sec<<','<<s.resets<<','<<reference_points<<','<<reference_planes<<std::endl;
    if(!temporal_initialization_log_) throw std::runtime_error("Cannot write temporal initialization diagnostics");
  };
  if(initializing) {
    if(!temporal_initialization_log_.is_open()) {
      const auto directory=debug_log_dir.empty() ? std::filesystem::path(ROOT_DIR)/"Log" : std::filesystem::path(debug_log_dir);
      std::filesystem::create_directories(directory);
      temporal_initialization_log_.open(directory/"temporal_initialization.csv");
      if(!temporal_initialization_log_) throw std::runtime_error("Cannot open temporal initialization diagnostics");
      temporal_initialization_log_<<"time,mode,status,gate_applied,scans,points,planes,normal_eigenvalue_0,normal_eigenvalue_1,normal_eigenvalue_2,duration_sec,resets,reference_points,temporal_reference_planes\n"<<std::setprecision(12);
    }
    if(legacy_initialization) {
      if(voxelmap_manager->config_setting_.correspondence_mode_!="voxel_plane")
        throw std::runtime_error("Temporal legacy_voxel initialization requires lio.correspondence_mode=voxel_plane");
      // Identical preprocessing, accumulation, and readiness gate to handleLIO.
      // No IEKF observation update is performed, so no measurement is reused
      // as both a posterior-state prior and a temporal factor.
      downSizeFilterSurf.setInputCloud(feats_undistort);
      // Allocate with this translation unit's Eigen alignment. Binary PCL can
      // use a different aligned allocator than our -march=native build; letting
      // it allocate this vector can crash when we later destroy the cloud.
      // VoxelGrid produces at most one output per input, so it cannot grow past
      // this capacity. This changes ownership of allocation, not the filtering.
      feats_down_body->reserve(feats_undistort->size());
      downSizeFilterSurf.filter(*feats_down_body);
      feats_down_size=feats_down_body->size();
      voxelmap_manager->feats_down_body_=feats_down_body;
      transformLidar(_state.rot_end,_state.pos_end,feats_down_body,feats_down_world);
      voxelmap_manager->feats_down_world_=feats_down_world;
      voxelmap_manager->feats_down_size_=feats_down_size;
      std::vector<pointWithVar> accepted_seed;
      lidar_map_inited=voxelmap_manager->TryInitializeVoxelMap(time,&accepted_seed);
      if(!lidar_map_inited) {
        logInitialization(false,0,0);
        trimImu();
        return;
      }
      if(accepted_seed.empty()) throw std::runtime_error("Temporal initialization: READY without seed geometry");
      initialization_reference.reserve(accepted_seed.size());
      for(const auto& p:accepted_seed)
        initialization_reference.push_back(_state.rot_end.transpose()*(p.point_w-_state.pos_end));
      // The temporal backend fits its usual planes to this initial geometry.
      // The legacy octree is no longer used after its readiness decision.
      for(auto& entry:voxelmap_manager->voxel_map_) delete entry.second;
      voxelmap_manager->voxel_map_.clear();
    }
  }
  if(!temporal_window_) {
    temporal_config_.gravity=_state.gravity;
    temporal_config_.lidar_to_imu_rotation=extR; temporal_config_.lidar_to_imu_translation=extT;
    temporal_config_.gyro_variance=p_imu->cov_gyr; temporal_config_.accel_variance=p_imu->cov_acc;
    temporal_config_.gyro_bias_variance=p_imu->cov_bias_gyr; temporal_config_.accel_bias_variance=p_imu->cov_bias_acc;
    temporal_window_=std::make_unique<temporal_lio::Window>(temporal_config_);
    const auto directory=debug_log_dir.empty() ? std::filesystem::path(ROOT_DIR)/"Log" : std::filesystem::path(debug_log_dir);
    std::filesystem::create_directories(directory);
    temporal_log_.open(directory/"temporal_window.csv"); temporal_states_log_.open(directory/"temporal_states.csv");
    if(!temporal_log_ || !temporal_states_log_) throw std::runtime_error("Cannot open temporal LIO diagnostics");
    temporal_log_<<"time,status,nodes,span,matches,imu_edges,marginalized,iterations,cost_before,cost_after,imu_rms,lidar_rms,max_imu_gap,solve_ms,qw,qx,qy,qz,px,py,pz,vx,vy,vz,bgx,bgy,bgz,bax,bay,baz";
    for(int i=0;i<15;++i) temporal_log_<<",correction_"<<i;
    for(const auto& prefix:{"cov_","marginal_information_"})
      for(int i=0;i<15;++i) for(int j=0;j<15;++j) temporal_log_<<','<<prefix<<i<<'_'<<j;
    temporal_log_<<",structure_planes,structure_rms_deg,structure_information,map_planes,map_matches,landmarks,saturated_imu_intervals";
    temporal_log_<<'\n'<<std::setprecision(12);
    temporal_states_log_<<"time,node_id,node_time,keyframe,qw,qx,qy,qz,px,py,pz,vx,vy,vz,bgx,bgy,bgz,bax,bay,baz\n"<<std::setprecision(12);
  }
  std::vector<temporal_lio::Return> points;
  points.reserve(LidarMeasures.pcl_proc_cur->size());
  for(const auto& p:*LidarMeasures.pcl_proc_cur) {
    if(!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) continue;
    points.push_back({V3D(p.x,p.y,p.z),temporal_scan_begin_time_+p.curvature*0.001,p.intensity});
  }
  auto imu=temporal_imu_;
  if(!(p_imu->IMU_mean_acc_norm>0)) throw std::runtime_error("Temporal LIO: invalid IMU scale");
  for(auto& p:imu) p.accel*=G_m_s2/p_imu->IMU_mean_acc_norm;
  temporal_lio::M15 covariance;
  for(int i=0;i<15;++i) for(int j=0;j<15;++j) covariance(i,j)=_state.cov(state_index[i],state_index[j]);
  const auto report=temporal_window_->add(time,points,imu,temporalState(_state),covariance,initialization_reference,
      temporal_visual_reprojection_ && img_en);
  if(initializing) {
    const auto& first=temporal_window_->nodes().front();
    logInitialization(true,legacy_initialization ? first.initialization_reference.size() : first.points.size(),first.planes.size());
    RCLCPP_INFO(node_->get_logger(),"Temporal initialization %s READY at %.3f s: %zu reference points, %zu temporal planes",
      temporal_initialization_mode_.c_str(),time-_first_lidar_time,
      legacy_initialization ? first.initialization_reference.size() : first.points.size(),first.planes.size());
  }
  applyTemporalState(report);
  lidar_map_inited=true;
  const auto& s=temporal_window_->state();
  const double relative=time-_first_lidar_time;
  temporal_log_<<relative<<','<<report.status<<','<<report.nodes<<','<<report.span<<','<<report.matches<<','<<report.imu_edges<<','<<report.marginalized<<','<<report.iterations
    <<','<<report.cost_before<<','<<report.cost_after<<','<<report.imu_rms<<','<<report.lidar_rms<<','<<report.max_imu_gap<<','<<report.solve_ms;
  writeState(temporal_log_,s);
  for(int i=0;i<15;++i) temporal_log_<<','<<report.correction[i];
  for(const auto* m:{&report.covariance,&report.information})
    for(int i=0;i<15;++i) for(int j=0;j<15;++j) temporal_log_<<','<<(*m)(i,j);
  int structure_planes=0;double structure_error=0,structure_information=0;
  for(const auto& observation:temporal_window_->nodes().back().structure) {
    if(observation.weight<=0) continue;
    const double angle=std::acos(std::clamp(observation.axis.dot(s.rotation*observation.plane.normal),-1.,1.));
    structure_error+=observation.weight*angle*angle;structure_information+=observation.weight;++structure_planes;
  }
  temporal_log_<<','<<structure_planes<<','<<std::sqrt(structure_error/std::max(1.,structure_information))*180./M_PI
    <<','<<structure_information<<','<<report.map_planes<<','<<report.map_matches<<','<<report.landmarks<<','<<report.saturated_imu_intervals;
  temporal_log_<<std::endl;
  for(const auto& n:temporal_window_->nodes()) {
    temporal_states_log_<<relative<<','<<n.id<<','<<n.time-_first_lidar_time<<','<<n.keyframe;
    writeState(temporal_states_log_,n.state); temporal_states_log_<<'\n';
  }
  temporal_states_log_.flush();
  if(!temporal_log_ || !temporal_states_log_) throw std::runtime_error("Cannot write temporal LIO diagnostics");
  if(imu_prop_enable) {
    ekf_finish_once=true; latest_ekf_state=_state; latest_ekf_time=time; state_update_flg=true;
  }
  euler_cur=RotMtoEuler(_state.rot_end);
  geoQuat=tf::createQuaternionMsgFromRollPitchYaw(euler_cur(0),euler_cur(1),euler_cur(2));
  if(!img_en) publish_odometry(pubOdomAftMapped);
  // Publish freshly re-deskewed returns, not the predictor's old deskew result.
  feats_undistort->clear(); pcl_w_wait_pub->clear();
  const auto& latest=temporal_window_->nodes().back();
  for(size_t i=0;i<latest.points.size();++i) {
    const V3D body=extR.transpose()*(latest.points[i]-extT);
    const V3D world=s.rotation*latest.points[i]+s.position;
    PointType p{}; p.x=body.x();p.y=body.y();p.z=body.z();p.intensity=latest.raw[i].intensity;
    feats_undistort->push_back(p); p.x=world.x();p.y=world.y();p.z=world.z();pcl_w_wait_pub->push_back(p);
  }
  // Preserve the existing display contract: /Laser_map accumulates the whole
  // run at the published poses. It is independent of the estimator's window
  // and is not a retrospectively optimized map.
  if(!img_en) {
    update_and_publish_map(pcl_w_wait_pub);
    publish_frame_world(pubLaserCloudFullRes,vio_manager);
    publishTemporalPath(time);
  }
  if(pose_output_en && !img_en) {
    std::ofstream evo(std::string(ROOT_DIR)+"Log/result/"+seq_name+"_temporal.txt",frame_num ? std::ios::app : std::ios::out);
    if(!evo) throw std::runtime_error("Cannot open temporal trajectory output");
    Eigen::Quaterniond q(s.rotation);
    evo<<std::fixed<<std::setprecision(9)<<time<<' '<<s.position.transpose()<<' '<<q.x()<<' '<<q.y()<<' '<<q.z()<<' '<<q.w()<<'\n';
  }
  fout_out<<std::setprecision(12)<<relative<<' '<<euler_cur.transpose()*57.3<<' '<<s.position.transpose()<<' '<<s.velocity.transpose()<<' '
    <<s.gyro_bias.transpose()<<' '<<s.accel_bias.transpose()<<' '<<V3D(_state.inv_expo_time,0,0).transpose()<<' '<<points.size()<<std::endl;
  ++frame_num;
  if(frame_num%100==0) RCLCPP_INFO(node_->get_logger(),"Temporal LIO t=%.1f: %d nodes, %.2f s, %d matches, %.1f ms/scan",relative,report.nodes,report.span,report.matches,report.solve_ms);
  trimImu();
}

void LIVMapper::applyTemporalState(const temporal_lio::Report& report) {
  const auto& s=temporal_window_->state();
  _state.rot_end=s.rotation; _state.pos_end=s.position; _state.vel_end=s.velocity;
  _state.bias_g=s.gyro_bias; _state.bias_a=s.accel_bias;
  // Gravity is fixed; exposure is a separate visual nuisance. Only the 15 estimated
  // coordinates and their joint marginal covariance are transferred.
  for(int i=0;i<15;++i) {
    _state.cov.row(state_index[i]).setZero(); _state.cov.col(state_index[i]).setZero();
  }
  for(int i=0;i<15;++i) for(int j=0;j<15;++j) _state.cov(state_index[i],state_index[j])=report.covariance(i,j);
}

void LIVMapper::publishTemporalPath(double time) {
  temporal_trajectory_.update(temporal_window_->nodes());
  path.poses.clear();
  path.poses.reserve(temporal_trajectory_.poses().size());
  for(const auto& n:temporal_trajectory_.poses()) {
    geometry_msgs::PoseStamped p;p.header.frame_id="camera_init";p.header.stamp=fast_livo2::stampFromSec(n.time);
    p.pose.position.x=n.position.x();p.pose.position.y=n.position.y();p.pose.position.z=n.position.z();
    Eigen::Quaterniond q(n.rotation);p.pose.orientation.w=q.w();p.pose.orientation.x=q.x();p.pose.orientation.y=q.y();p.pose.orientation.z=q.z();
    path.poses.push_back(p);
  }
  path.header.stamp=fast_livo2::stampFromSec(time);pubPath.publish(path);publish_mavros(mavros_pose_publisher);
}

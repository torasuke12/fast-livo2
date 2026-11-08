#include "vio.h"
#include "temporal_scan.h"
#include <iostream>
#include <stdexcept>
#include <limits>

static void checkFeatureRetrieval(const V3D& world_point) {
  vk::PinholeCamera camera(640,480,600,600,320,240);
  StatesGroup state;state.inv_expo_time=1.;
  VIOManager vio;vio.cam=&camera;vio.state=&state;vio.state_propagat=&state;
  vio.Rli=vio.Rcl=M3D::Identity();vio.Pli=vio.Pcl=V3D::Zero();
  vio.grid_size=28;vio.grid_n_height=17;vio.patch_size=8;vio.patch_pyrimid_level=4;
  vio.normal_en=true;vio.raycast_en=false;vio.outlier_threshold=500.;
  vio.initializeVIO();vio.resetGrid();
  cv::Mat image(480,640,CV_8UC1,cv::Scalar(128));
  vio.new_frame_=std::make_unique<Frame>(&camera,image);
  vio.new_frame_->T_f_w_=SE3(M3D::Identity(),V3D(0,0,5)-world_point);
  auto* point=new VisualPoint(world_point);
  point->normal_=point->previous_normal_=V3D::UnitZ();
  point->is_normal_initialized_=true;point->covariance_=M3D::Identity();
  auto* patch=new float[64];std::fill(patch,patch+64,128.f);
  auto* feature=new Feature(point,patch,V2D(320,240),V3D::UnitZ(),vio.new_frame_->T_f_w_,0);
  feature->img_=image;feature->inv_expo_time_=1.;feature->id_=0;
  point->addFrameRef(feature);vio.insertPointIntoVoxelMap(point);
  std::vector<pointWithVar> scan(1);scan[0].point_w=world_point;
  vio.retrieveFromVisualSparseMap(image,scan,{});
  if(vio.total_points!=1 || vio.visual_submap->voxel_points.front()!=point)
    throw std::runtime_error("a visible feature at the scan location was lost by voxel lookup: " +
                             std::to_string(world_point.x())+","+std::to_string(world_point.y())+","+std::to_string(world_point.z()));
  // A state update can move an already retrieved point behind the camera.
  vio.new_frame_->T_f_w_=SE3(M3D::Identity(),V3D(0,0,-5)-world_point);
  vio.updateVisualMapPoints(image);
  if(point->obs_.size()!=1) throw std::runtime_error("invisible point acquired a new reference patch");
}

static void checkBehindCameraCandidates() {
  vk::PinholeCamera camera(640,480,600,600,320,240);
  StatesGroup state;state.inv_expo_time=1.;
  VIOManager vio;vio.cam=&camera;vio.state=&state;vio.state_propagat=&state;
  vio.Rli=vio.Rcl=M3D::Identity();vio.Pli=vio.Pcl=V3D::Zero();
  vio.grid_size=28;vio.grid_n_height=17;vio.patch_size=8;vio.patch_pyrimid_level=4;
  vio.normal_en=true;vio.raycast_en=false;vio.outlier_threshold=500.;
  vio.initializeVIO();vio.resetGrid();
  cv::Mat image(480,640,CV_8UC1);
  for(int y=0;y<480;++y) for(int x=0;x<640;++x)
    image.at<unsigned char>(y,x)=((x/8+y/8)%2)*255;
  vio.new_frame_=std::make_unique<Frame>(&camera,image);
  vio.new_frame_->T_f_w_=SE3(M3D::Identity(),V3D::Zero());
  std::vector<pointWithVar> scan(11);
  for(auto& p:scan) {p.point_w=V3D(0,0,-5);p.normal=V3D::UnitZ();p.var=M3D::Identity();}
  scan[0].point_w.z()=0;
  scan[1].point_w.x()=std::numeric_limits<double>::quiet_NaN();
  scan[2].point_w=V3D(1e100,0,1e-5);
  // Negative depth still projects to the image center. It must not create a
  // feature or steal this grid cell from a real point in front of the camera.
  vio.generateVisualMapPoints(image,scan);
  if(!vio.feat_map.empty()) throw std::runtime_error("point behind camera created an image feature");
  scan.back().point_w.z()=5;
  vio.generateVisualMapPoints(image,scan);
  if(vio.feat_map.size()!=1 || vio.feat_map.begin()->second->voxel_points.front()->pos_.z()<=0)
    throw std::runtime_error("behind-camera candidate suppressed valid image feature");
}

static void checkReferenceCapture() {
  vk::PinholeCamera camera(640,480,600,600,320,240);
  StatesGroup state;
  VIOManager vio;vio.cam=&camera;vio.state=&state;vio.state_propagat=&state;
  vio.Rli=vio.Rcl=M3D::Identity();vio.Pli=vio.Pcl=V3D::Zero();
  vio.grid_size=28;vio.grid_n_height=17;vio.patch_size=8;vio.patch_pyrimid_level=4;
  vio.img_point_cov=1.;vio.initializeVIO();vio.temporal_joint_reference=true;
  temporal_lio::Node reference,current;reference.id=7;current.id=8;
  vio.synchronizeTemporalMap({reference,current});
  for(uint64_t id:{7,8,99}) {
    auto* point=new VisualPoint(V3D(1,2,5));point->temporal_anchor_=id;
    vio.insertPointIntoVoxelMap(point);vio.visual_submap->voxel_points.push_back(point);
  }
  MatrixXd j=MatrixXd::Random(192,6);VectorXd r=VectorXd::Random(192);
  vio.captureTemporalObservation(j,r);
  const auto& observation=vio.temporal_observation;
  if(observation.references.size()!=1 || observation.references.front().node!=7)
    throw std::runtime_error("image capture retained current or expired point anchors");
  MatrixXd combined(observation.residual.size(),12);
  combined<<observation.jacobian,observation.references.front().jacobian;
  MatrixXd translation=MatrixXd::Zero(12,3);
  translation.block<3,3>(3,0).setIdentity();translation.block<3,3>(9,0).setIdentity();
  if((combined*translation).norm()>1e-10)
    throw std::runtime_error("captured reference image measured global translation");
  // Only the first patch has a retained, distinct reference pose.
  if((observation.jacobian.transpose()*observation.jacobian-j.topRows(64).transpose()*j.topRows(64)).norm()>1e-9)
    throw std::runtime_error("invalid anchor pixels contributed image information");
  vio.synchronizeTemporalMap({});
  vio.resetGrid();
  cv::Mat image(480,640,CV_8UC1,cv::Scalar(128));
  std::vector<pointWithVar> empty_scan;
  vio.retrieveFromVisualSparseMap(image,empty_scan,{});
  if(!vio.visual_submap->voxel_points.empty() || vio.total_points!=0)
    throw std::runtime_error("empty visual map left stale point pointers for plotting");
}

int main() {
  try {
    checkReferenceCapture();
    checkBehindCameraCandidates();
    // Translating the same image/point scene across the world origin must not
    // change feature retrieval. Include exact negative voxel boundaries.
    for(const V3D& point:{V3D(.23,.23,.23),V3D(-.23,-.23,-.23),
                         V3D(.23,-.23,.23),V3D(-.5,-1.,-1.5),V3D::Zero().eval()})
      checkFeatureRetrieval(point);
    PointCloudXYZI pending,current;
    for(int k=0;k<10;++k) {PointType p{};p.curvature=10*k;p.intensity=k;pending.push_back(p);}
    int seen=0;
    for(int cut=0;cut<4;++cut) {
      temporal_lio::splitPendingScan(pending,current,30);
      for(const auto& point:current) {
        if(point.intensity!=seen++ || std::abs(point.curvature+30*cut-10*point.intensity)>1e-6)
          throw std::runtime_error("image cut loses, duplicates or retimes pending LiDAR returns");
        if(point.curvature>=30) throw std::runtime_error("future return included before image time");
      }
    }
    if(seen!=10 || !pending.empty()) throw std::runtime_error("image cuts did not preserve all returns");
    VIOManager vio;temporal_lio::Node node;node.id=7;node.state.position=V3D(1,2,3);
    vio.synchronizeTemporalMap({node});
    auto* point=new VisualPoint(V3D(2,4,6));point->temporal_anchor_=7;
    point->normal_=point->previous_normal_=V3D::UnitZ();point->covariance_=M3D::Identity();
    auto* feature=new Feature(point,new float[1],V2D::Zero(),V3D::UnitZ(),SE3(M3D::Identity(),V3D(-1,-2,-3)),0);
    feature->temporal_anchor_=7;point->addFrameRef(feature);vio.insertPointIntoVoxelMap(point);
    const V3D camera_point=feature->T_f_w_*point->pos_;
    node.state.rotation=temporal_lio::expSO3(V3D(.1,.2,.3));node.state.position=V3D(3,1,5);
    vio.synchronizeTemporalMap({node});
    if((feature->T_f_w_*point->pos_-camera_point).norm()>1e-10)
      throw std::runtime_error("reference image and point anchors diverged");
    if((point->pos_-(node.state.rotation*V3D(1,2,3)+node.state.position)).norm()>1e-10)
      throw std::runtime_error("visual point did not follow retained pose");
    const V3D saved=point->pos_;vio.synchronizeTemporalMap({node});
    if((point->pos_-saved).norm()>1e-10) throw std::runtime_error("pose correction applied twice");
    temporal_lio::Node next;next.id=8;vio.synchronizeTemporalMap({next});
    if((point->pos_-saved).norm()>1e-10) throw std::runtime_error("marginalized visual anchor changed");
    vio.temporal_joint_reference=true;vio.synchronizeTemporalMap({next});
    if(!vio.feat_map.empty()) throw std::runtime_error("joint image mode retained an expired point anchor");
    std::cout<<"temporal visual map anchor tests passed\n";
  } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}

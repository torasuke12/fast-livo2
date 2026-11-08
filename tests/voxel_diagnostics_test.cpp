#include "voxel_map.h"
#include <Eigen/Geometry>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>

namespace d = fast_livo2::diagnostics;
using V = Eigen::Vector3d;
using M = Eigen::Matrix3d;
void require(bool value, const char *message) { if (!value) throw std::runtime_error(message); }

VoxelMapConfig config()
{
  VoxelMapConfig c{};
  c.correspondence_mode_="voxel_plane"; c.max_voxel_size_=.5; c.max_layer_=2;
  c.layer_init_num_={5,5,5}; c.max_points_num_=100; c.planner_threshold_=.0001;
  c.dept_err_=.02; c.beam_err_=.05; c.max_state_cov_for_matching_=.01;
  c.neighbor_voxel_radius_=1; c.sigma_num_=3; c.max_iterations_=2;
  c.xicp_.enough_information=10; c.xicp_.insufficient_information=6; c.xicp_.minimal_information=1;
  c.diagnostics_.detail_start_sec=0; c.diagnostics_.detail_end_sec=360;
  c.diagnostics_.parameters["lio.sigma_num"]="3";
  return c;
}

struct Fixture
{
  std::unordered_map<VOXEL_LOCATION,VoxelOctoTree*> empty;
  std::unique_ptr<VoxelMapManager> m;
  explicit Fixture(VoxelMapConfig c):m(new VoxelMapManager(c,empty))
  { m->extR_=Eigen::AngleAxisd(.21,V::UnitY()).toRotationMatrix();m->extT_=V(.1,-.07,.05); }
  ~Fixture() { for(auto &v:m->voxel_map_)delete v.second; }
  VoxelOctoTree *plane(const V &center, int axis)
  {
    VOXEL_LOCATION key(std::floor(center.x()/.5),std::floor(center.y()/.5),std::floor(center.z()/.5));
    auto *node=new VoxelOctoTree(2,0,5,100,.0001);node->layer_init_num_={5,5,5};node->quater_length_=.125;
    node->voxel_center_[0]=(key.x+.5)*.5;node->voxel_center_[1]=(key.y+.5)*.5;node->voxel_center_[2]=(key.z+.5)*.5;
    std::vector<pointWithVar> points;
    for(int i=-2;i<=2;++i)for(int j=-2;j<=2;++j)
    {
      pointWithVar p;p.point_w=center;p.point_w[(axis+1)%3]+=.03*i;p.point_w[(axis+2)%3]+=.03*j;
      p.var=M::Identity()*.0004;points.push_back(p);
    }
    node->init_plane(points,node->plane_ptr_);node->init_octo_=true;
    require(node->plane_ptr_->is_plane_ && node->plane_ptr_->plane_var_.allFinite(),"plane fixture");
    m->voxel_map_[key]=node;return node;
  }
  void query(const V &world, const StatesGroup &prior)
  {
    const V b=m->extR_.transpose()*(prior.rot_end.transpose()*(world-prior.pos_end)-m->extT_);
    PointType p{};p.x=b.x();p.y=b.y();p.z=b.z();m->feats_down_body_->push_back(p);
    m->feats_down_size_=m->feats_down_body_->size();
  }
};

StatesGroup integrated(const std::filesystem::path &out, bool diagnostics, bool xicp)
{
  auto c=config();c.xicp_.enabled=xicp;
  c.diagnostics_.enabled=diagnostics;c.diagnostics_.geometry_enabled=diagnostics;
  c.diagnostics_.detail_enabled=diagnostics;c.diagnostics_.detail_final_only=true;
  c.diagnostics_.xicp_enabled=xicp;c.diagnostics_.output_dir=out.string();
  Fixture f(c);StatesGroup prior;prior.cov.setIdentity();prior.cov*=.0001;
  prior.cov(7,3)=prior.cov(3,7)=.00002;prior.cov(13,3)=prior.cov(3,13)=.00001;
  prior.rot_end=Eigen::AngleAxisd(.15,V::UnitZ()).toRotationMatrix();prior.pos_end=V(.1,.2,-.1);
  prior.vel_end=V(1,0,0);prior.bias_a=V(.1,.02,.03);f.m->state_=prior;
  for(int axis=0;axis<3;++axis)
  {
    V center(2.25+2*axis,1.25,1.25);center[axis]+=(axis ? 1:0);
    auto *node=f.plane(center,axis);if(axis==1)node->update_enable_=false;
    for(int i=-2;i<=2;++i)for(int j=-2;j<=2;++j)
    {
      V q=center;q[axis]+=.008;q[(axis+1)%3]+=.02*i;q[(axis+2)%3]+=.02*j;
      f.query(q,prior);
    }
  }
  f.query(V(2.45,1.25,1.25),prior); // residual rejection
  f.query(V(2.25,1.60,1.25),prior); // extent rejection
  f.query(V(20,10,5),prior);        // no voxel
  auto *no_plane=f.plane(V(8.25,3.25,2.25),0);no_plane->plane_ptr_->is_plane_=false;
  f.query(V(8.25,3.25,2.25),prior);
  auto *bad=f.plane(V(10.25,3.25,2.25),0);bad->plane_ptr_->plane_var_(0,0)=std::numeric_limits<double>::quiet_NaN();
  f.query(V(10.26,3.25,2.25),prior);
  f.m->StateEstimation(prior,205.1);
  require(f.m->effct_feat_num_==75,"voxel accepted set must exclude invalid/missing/rejected planes");
  require(f.m->state_.cov.allFinite() && f.m->state_.vel_end.allFinite(),"finite state/covariance");
  require((f.m->state_.pos_end-prior.pos_end).norm()>1e-5,"fixture exercises an actual update");
  return f.m->state_;
}

void varianceAndRejections()
{
  Fixture f(config());const V center(2.25,1.25,1.25);auto *node=f.plane(center,0);
  pointWithVar pv;pv.point_w=center+V(.01,0,0);pv.point_b=pv.point_w;
  pv.var=M::Identity()*.0004;pv.body_var=pv.var;
  PointToPlane match;bool success=false;double probability=0;d::GeometryPoint r;
  f.m->build_single_residual(pv,node,0,success,probability,match,&r);
  require(success && r.status==d::MatchStatus::Accepted,"variance gate accepted");
  const double variance=r.voxel.gate_variance;
  node->plane_ptr_->plane_var_*=10;success=false;probability=0;r={};
  f.m->build_single_residual(pv,node,0,success,probability,match,&r);
  require(success && r.voxel.gate_variance>variance,"larger plane uncertainty reaches residual variance");
  node->plane_ptr_->plane_var_.setZero();pv.var.setZero();success=false;probability=0;r={};
  f.m->build_single_residual(pv,node,0,success,probability,match,&r);
  require(!success && r.status==d::MatchStatus::InvalidVariance,"zero variance rejected, no uninitialized match");
  pv.point_w=center;pv.var=M::Identity()*.0004;success=false;probability=0;r={};
  f.m->build_single_residual(pv,node,0,success,probability,match,&r);
  require(success && r.voxel.tangential_distance==0,"plane center roundoff");
  // A later rejected candidate must not overwrite the selected plane record.
  const auto id=r.voxel.plane_id;auto *far=f.plane(V(2.25,1.75,1.25),0);
  f.m->build_single_residual(pv,far,0,success,probability,match,&r);
  require(r.voxel.plane_id==id && r.status==d::MatchStatus::Accepted,"selected record survives later rejection");
}

void noMatchesRollback(const std::filesystem::path &out, bool xicp)
{
  auto c=config();c.xicp_.enabled=xicp;c.diagnostics_.enabled=true;c.diagnostics_.geometry_enabled=true;
  c.diagnostics_.detail_enabled=true;c.diagnostics_.detail_final_only=true;c.diagnostics_.output_dir=out.string();
  Fixture f(c);f.plane(V(2.25,1.25,1.25),0);
  StatesGroup prior;prior.vel_end=V(1,2,3);prior.bias_a=V(.1,.2,.3);prior.cov.setIdentity();prior.cov*=.123;
  f.query(V(20,10,5),prior);f.m->state_=prior;f.m->state_.pos_end.x()=.05;f.m->state_.vel_end.setZero();
  f.m->StateEstimation(prior,2);
  require(f.m->effct_feat_num_==0 && (f.m->state_-prior).norm()==0 && (f.m->state_.cov-prior.cov).norm()==0,
          "voxel no-match restores whole IMU prior, even without XICP");
}

int main(int argc,char **argv)
{
  try
  {
    rclcpp::init(argc,argv);
    rclcpp::NodeOptions opts;opts.parameter_overrides({rclcpp::Parameter("lio.correspondence_mode","voxel_plane"),
      rclcpp::Parameter("lio.diagnostics.enabled",true),rclcpp::Parameter("lio.diagnostics.detail.enabled",true),
      rclcpp::Parameter("lio.diagnostics.geometry.enabled",true)});
    auto node=std::make_shared<rclcpp::Node>("voxel_diagnostic_config_test",opts);VoxelMapConfig loaded;
    loadVoxelConfig(node,loaded);require(loaded.diagnostics_.correspondence_mode=="voxel_plane","config allows voxel geometry");
    const char *output=std::getenv("VOXEL_TEST_OUTPUT");char temporary[]="/tmp/fast-livo2-voxel-test-XXXXXX";
    const auto root=output ? std::filesystem::path(output) : std::filesystem::path(mkdtemp(temporary));
    varianceAndRejections();
    for(bool xicp:{false,true})
    {
      auto a=integrated(root/(xicp ? "xicp":"baseline"),true,xicp);
      const auto b=integrated(root/"quiet",false,xicp);
      require((a-b).norm()==0 && (a.cov-b.cov).norm()==0,"diagnostics must not change estimation");
      noMatchesRollback(root/(xicp ? "empty-xicp":"empty-baseline"),xicp);
    }
    if(!output)std::filesystem::remove_all(root);
    rclcpp::shutdown();std::cout<<"Voxel-plane diagnostics tests passed\n";return 0;
  }
  catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}
}

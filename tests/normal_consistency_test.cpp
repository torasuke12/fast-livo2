#include "normal_consistency.h"
#include "voxel_map.h"
#include "recorded_normal_fixture.h"
#include <Eigen/Geometry>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>

namespace nc = fast_livo2::normal_consistency;
using V = Eigen::Vector3d;
using M = Eigen::Matrix3d;
void require(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
PointType point(const V &v) { PointType p{}; p.x=v.x(); p.y=v.y(); p.z=v.z(); return p; }
M matrix(const double *v) { return Eigen::Map<const Eigen::Matrix<double,3,3,Eigen::RowMajor>>(v); }

void pcaAndComparison()
{
  nc::Config c; c.validate();
  std::vector<V> patch;
  for (int i=0; i<4; ++i) for (int j=0; j<5; ++j) patch.emplace_back(.1*i,.1*j,2.0);
  auto n=nc::fit(patch,.5,c);
  require(n.usable() && std::abs(n.normal.z()) > .999999, "centered planar PCA");
  auto same=nc::compare(n,-V::UnitZ(),M::Identity(),M::Identity(),c);
  require(same.valid && !same.mismatch && same.angle_deg < 1e-6, "sign invariance");
  const M R=Eigen::AngleAxisd(.7,V::UnitX()).toRotationMatrix();
  const M E=Eigen::AngleAxisd(-.6,V::UnitY()).toRotationMatrix();
  require(!nc::compare(n,R*E*n.normal,R,E,c).mismatch, "pose and extrinsic composition");
  for (const double angle : {29.999,30.0,30.001,90.0})
  {
    const V target=Eigen::AngleAxisd(angle*M_PI/180.,V::UnitY())*V::UnitZ();
    require(nc::compare(n,target,M::Identity(),M::Identity(),c).mismatch == (angle>30), "angle boundary");
  }
  require(nc::fit(patch,1.0,c).quality==nc::Quality::NeighborDistance,"radius boundary");
  patch.pop_back(); require(!nc::fit(patch,.5,c).usable(),"insufficient neighbors");
  patch.resize(20,V::Zero());
  for(int i=0;i<20;++i) patch[i]=V(.01*i,0,0);
  require(nc::fit(patch,.5,c).quality==nc::Quality::Degenerate,"line rejected as source normal");
  for(int i=0;i<4;++i)for(int j=0;j<5;++j)patch[i*5+j]=V(.1*i,.00001*j,0);
  require(nc::fit(patch,.5,c).quality==nc::Quality::Narrow,"narrow patch");
  for(int i=0;i<20;++i)patch[i]=V(.1*(i%2),.1*((i/2)%2),.1*((i/4)%2));
  require(nc::fit(patch,.5,c).quality==nc::Quality::NonPlanar,"nonplanar patch");
  patch.assign(20,V::Ones());require(!nc::fit(patch,0,c).usable(),"duplicates");
  patch[0].x()=std::numeric_limits<double>::quiet_NaN();
  require(!nc::fit(patch,.5,c).usable(),"NaN neighbor");
  pcl::PointCloud<PointType> cloud;
  for(const auto &p:patch) cloud.push_back(point(p));
  const auto normals=nc::estimate(cloud,c);
  require(normals[0].quality==nc::Quality::InvalidPoint && !normals[1].usable(),"finite source search");
  require(!nc::compare(normals[1],V::UnitX(),R,E,c).valid,"unavailable normal skips gate");
  for (int i=0;i<6;++i)
  {
    auto bad=c;
    if(i==0) bad.neighbors=2;
    if(i==1) bad.max_neighbor_distance=0;
    if(i==2) bad.max_planarity_ratio=1;
    if(i==3) bad.min_spread_ratio=0;
    if(i==4) bad.max_angle_deg=91;
    if(i==5) bad.max_angle_deg=std::numeric_limits<double>::quiet_NaN();
    bool threw=false;try{bad.validate();}catch(const std::invalid_argument&){threw=true;}
    require(threw,"invalid normal config accepted");
  }
}

VoxelMapConfig config()
{
  VoxelMapConfig c{};
  c.correspondence_mode_="fastlio_ikdtree";c.max_iterations_=2;
  c.normal_consistency_.enabled=true;
  c.fastlio_cube_side_length_=1000;c.fastlio_detection_range_=100;
  c.fastlio_filter_size_map_=.5;c.dept_err_=.02;c.beam_err_=.05;
  c.max_state_cov_for_matching_=.01;
  return c;
}
void recordedAndCached()
{
  using namespace recorded_normal_fixture;
  auto c=config();std::unordered_map<VOXEL_LOCATION,VoxelOctoTree*> empty;
  auto m=std::make_unique<VoxelMapManager>(c,empty);
  m->extR_=matrix(extrinsic_R);m->extT_=Eigen::Map<const V>(extrinsic_t);
  m->state_.rot_end=matrix(linearization_R);
  for(int i=0;i<20;++i)m->feats_down_body_->push_back(point(Eigen::Map<const V>(source_points+3*i)));
  m->PrepareSourceNormals();
  const auto source=nc::estimate(*m->feats_down_body_,c.normal_consistency_);
  require(source[0].usable(),"recorded source normal usable");
  PointVector map;
  for(int i=0;i<5;++i)map.push_back(point(Eigen::Map<const V>(map_points+3*i)));
  m->fastlio_ikdtree_.Build(map);
  // Source index 0 is the original query in this small fixture.
  std::vector<pointWithVar> queries(1);queries[0].point_b=Eigen::Map<const V>(query_lidar);
  queries[0].point_w=Eigen::Map<const V>(query_world);queries[0].body_var=M::Identity();
  m->fastlio_nearest_points_.resize(1);m->fastlio_neighbor_valid_.resize(1);
  std::vector<PointToPlane> matches;
  m->config_setting_.normal_consistency_.enabled=false;
  m->BuildResidualListFastLio(queries,matches,true);
  require(matches.size()==1,"recorded false match must pass old gate");
  auto comparison=nc::compare(source[0],matches[0].normal_,m->state_.rot_end,m->extR_,c.normal_consistency_);
  require(comparison.angle_deg>78 && comparison.angle_deg<80,"recorded approximately 79 degree mismatch");
  const V normal=matches[0].normal_;
  m->config_setting_.normal_consistency_.enabled=true;
  m->BuildResidualListFastLio(queries,matches,false,0);
  require(matches.empty(),"recorded cached mismatch rejected");
  m->state_.rot_end=Eigen::Quaterniond::FromTwoVectors(m->extR_*source[0].normal,normal).toRotationMatrix();
  m->BuildResidualListFastLio(queries,matches,false,1);
  require(matches.size()==1,"cached neighbors must reevaluate changed rotation");
  m->feats_down_body_->clear();m->feats_down_body_->push_back(point(Eigen::Map<const V>(query_lidar)));
  m->PrepareSourceNormals();
  m->state_.rot_end=matrix(linearization_R);
  m->BuildResidualListFastLio(queries,matches,false,2);
  require(matches.size()==1,"unavailable source normal preserves old gate");
  std::cout << "Recorded mismatch: " << comparison.angle_deg << " deg; OFF accepted, ON rejected\n";
}
void allRejectedRollback(bool xicp)
{
  auto c=config();c.xicp_.enabled=xicp;
  std::unordered_map<VOXEL_LOCATION,VoxelOctoTree*> empty;
  auto m=std::make_unique<VoxelMapManager>(c,empty);
  m->extR_.setIdentity();m->extT_.setZero();
  for(int i=0;i<4;++i)for(int j=0;j<5;++j)
    m->feats_down_body_->push_back(point(V(3+(i-1.5)*.04,(j-2)*.04,1)));
  m->feats_down_size_=m->feats_down_body_->size();
  PointVector map;
  for(const auto &v : {V(3,-.3,.7),V(3,.3,.7),V(3,-.3,1.3),V(3,.3,1.3),V(3,0,1)})map.push_back(point(v));
  m->fastlio_ikdtree_.Build(map);
  StatesGroup prior;prior.vel_end=V(1,2,3);prior.bias_a=V(.1,.2,.3);prior.bias_g=V(.01,.02,.03);
  prior.cov.setIdentity();prior.cov*=.123;prior.inv_expo_time=1.23;
  m->state_=prior;m->state_.pos_end.y()=.05;m->state_.vel_end.setZero();m->state_.cov*=2;
  m->StateEstimation(prior,1.0);
  require(m->ptpl_list_.empty() && m->effct_feat_num_==0,"all rejected set must reach filter/XICP");
  require((m->state_.rot_end-prior.rot_end).norm()==0 && (m->state_.pos_end-prior.pos_end).norm()==0 &&
          (m->state_.vel_end-prior.vel_end).norm()==0 && (m->state_.bias_a-prior.bias_a).norm()==0 &&
          (m->state_.bias_g-prior.bias_g).norm()==0 && (m->state_.gravity-prior.gravity).norm()==0 &&
          (m->state_.cov-prior.cov).norm()==0 && m->state_.inv_expo_time==prior.inv_expo_time,
          "all-rejected scan must return complete IMU prediction, also without XICP");
}
void diagnosticFilterSet(const std::filesystem::path &root, bool gate, bool xicp)
{
  auto c=config();c.normal_consistency_.enabled=gate;c.xicp_.enabled=xicp;
  c.diagnostics_.enabled=true;c.diagnostics_.detail_enabled=true;c.diagnostics_.detail_final_only=true;
  c.diagnostics_.geometry_enabled=true;c.diagnostics_.detail_start_sec=0;c.diagnostics_.detail_end_sec=360;
  c.diagnostics_.xicp_enabled=xicp;c.diagnostics_.output_dir=root.string();
  c.diagnostics_.parameters["lio.normal_consistency.enabled"]=gate ? "true" : "false";
  std::unordered_map<VOXEL_LOCATION,VoxelOctoTree*> empty;
  auto m=std::make_unique<VoxelMapManager>(c,empty);m->extR_.setIdentity();m->extT_.setZero();
  PointVector map;
  for(int group=0;group<2;++group)for(int i=0;i<4;++i)for(int j=0;j<5;++j)
  {
    const double a=(i-1.5)*.04, b=(j-2)*.04;
    m->feats_down_body_->push_back(point(group ? V(5,a,1+b) : V(3+a,b,1)));
    map.push_back(point(V(group ? 5 : 3,a,1+b)));
  }
  m->feats_down_size_=m->feats_down_body_->size();m->fastlio_ikdtree_.Build(map);
  StatesGroup prior;prior.cov.setIdentity();prior.cov*=.001;
  m->state_=prior;m->StateEstimation(prior,205.1);
  require(m->effct_feat_num_==(gate ? 20 : 40),"filter and XICP must use same filtered correspondence set");
  if(gate)for(const auto &match:m->ptpl_list_)require(match.point_b_.x()>4,"rejected set leaked into update");
}

int main(int argc,char **argv)
{
  try
  {
    rclcpp::init(argc,argv);pcaAndComparison();recordedAndCached();allRejectedRollback(false);allRejectedRollback(true);
    const char *output=std::getenv("NORMAL_TEST_OUTPUT");
    char temporary[]="/tmp/fast-livo2-normal-test-XXXXXX";
    const auto root=output ? std::filesystem::path(output) : std::filesystem::path(mkdtemp(temporary));
    for(bool gate:{false,true})for(bool xicp:{false,true})
      diagnosticFilterSet(root/(std::string(gate ? "on" : "off")+(xicp ? "-xicp" : "-baseline")),gate,xicp);
    if(!output)std::filesystem::remove_all(root);
    rclcpp::NodeOptions options;options.parameter_overrides({rclcpp::Parameter("lio.correspondence_mode","voxel_plane"),
      rclcpp::Parameter("lio.normal_consistency.enabled",true)});
    auto node=std::make_shared<rclcpp::Node>("normal_config_test",options);VoxelMapConfig c;
    bool threw=false;try{loadVoxelConfig(node,c);}catch(const std::invalid_argument&){threw=true;}
    require(threw,"gate on voxel-plane must fail startup");
    rclcpp::shutdown();std::cout << "Normal consistency tests passed\n";return 0;
  }
  catch(const std::exception &error){std::cerr << error.what() << '\n';return 1;}
}

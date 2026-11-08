#include "temporal_lio.h"
#include <iostream>
#include <stdexcept>

namespace temporal_lio {
struct DeskewDerivativeTest {
  static void pointJacobians() {
    Config c;c.lidar_to_imu_rotation=expSO3(V3(.1,-.2,.3));c.lidar_to_imu_translation=V3(.2,-.1,.05);
    State s;s.rotation=expSO3(V3(.4,.7,-1.8));s.position=V3(100,-80,20);s.velocity=V3(1,-.3,.2);
    s.gyro_bias=V3(.02,-.01,.015);s.accel_bias=V3(.1,-.04,.02);
    std::vector<ImuSample> imu;std::vector<Return> raw;
    for(int i=0;i<=20;++i) imu.push_back({.005*i,V3(.2+.03*i,-.4,.1),V3(.1*i,-.5,9.5)});
    for(int i=0;i<=17;++i) raw.push_back({V3(.2*i-2,1,-.4),.1*i/17,0});
    const auto original=deskew(raw,imu,.1,s,c);
    const V3 position=s.position;s.position+=V3(1e6,-2e6,3e6);
    const auto translated=deskew(raw,imu,.1,s,c);s.position=position;
    for(size_t i=0;i<raw.size();++i)
      if((original[i].array()!=translated[i].array()).any())
        throw std::runtime_error("World origin changed end-frame returns before voxel association");
    const auto j=deskewJacobians(raw,imu,.1,s,c);
    for(int k=0;k<15;++k) {
      V15 dx=V15::Zero();dx[k]=1e-5;
      const auto plus=deskew(raw,imu,.1,s.plus(dx),c),minus=deskew(raw,imu,.1,s.plus(-dx),c);
      for(size_t i=0;i<raw.size();++i)
        if((j[i].col(k)-(plus[i]-minus[i])/2e-5).norm()>2e-7)
          throw std::runtime_error("IMU-knot derivative disagrees with raw-return deskew");
    }
  }
  static void structuralGates() {
    Config c;c.structure_enabled=true;c.structure_min_planes=3;
    Window w(c);w.nodes_.emplace_back();
    for(int i=0;i<40;++i) {
      Plane p;const double a=i*M_PI/40;p.normal=V3(std::cos(a),std::sin(a),0);
      w.nodes_[0].planes[{i,0,0}]=p;
    }
    w.refreshStructure();
    if(w.structure_initialized_) throw std::runtime_error("Unstructured normals incorrectly set building heading");
    w.nodes_[0].planes.clear();
    for(int axis=0;axis<3;++axis) for(int i=0;i<6;++i) {
      Plane p;p.normal=V3::Unit(axis);w.nodes_[0].planes[{axis,i,0}]=p;
    }
    w.refreshStructure();
    if(!w.structure_initialized_) throw std::runtime_error("Building heading not initialized");
    w.nodes_[0].state.rotation=expSO3(V3(.015,-.02,.01));w.refreshStructure();
    auto cost=[&]() {double total=0;for(const auto& o:w.nodes_[0].structure)
      total+=o.weight*o.axis.cross(w.nodes_[0].state.rotation*o.plane.normal).squaredNorm();return total;};
    const double before=cost();const auto planes=w.nodes_[0].planes;
    for(const auto& entry:planes) {auto key=entry.first;key[2]=1;w.nodes_[0].planes.emplace(key,entry.second);}
    w.refreshStructure();
    if(before<1e-3 || std::abs(cost()-before)>1e-10)
      throw std::runtime_error("Duplicated plane patches multiplied structural information");
  }
  static void run(bool structure=false,bool persistent_map=false) {
    Config c;c.consistent_deskew=true;c.voxel_size=1.;c.sample_size=.04;
    c.persistent_map=persistent_map;c.plane_uncertainty=persistent_map;
    if(structure) {c.structure_enabled=true;c.structure_min_planes=3;c.voxel_size=.3;}
    Window window(c);
    for(int k=0;k<2;++k) {
      Node n;n.id=k;n.keyframe=true;n.time=.1*(k+1);
      n.state.rotation=expSO3(V3(.01,-.02,.015)*(k+1));
      n.state.velocity=V3(.3,-.2,.1);n.state.accel_bias=V3(.02,-.03,.01);
      n.state.gyro_bias=V3(.001,-.002,.003);
      for(int i=0;i<=20;++i) n.scan_imu.push_back({n.time-.1+i*.005,V3(.04,-.02,.1),V3(.2,-.1,9.8)});
      for(int axis=0;axis<3;++axis) for(int i=0;i<12;++i) for(int j=0;j<12;++j) {
        V3 p=V3::Zero();p[axis]=axis==2 ? -1.2 : 2.2;
        p[(axis+1)%3]=.1+i*.05;p[(axis+2)%3]=.1+j*.05;
        p+=k*V3(.01,-.015,.005);
        n.raw.push_back({p,n.time-.09+.007*((i+j)%10),0});
      }
      window.nodes_.push_back(n);window.prior_origins_.push_back(n.state);
    }
    window.prior_jacobian_=Matrix::Zero(1,30);window.prior_residual_=Vector::Zero(1);
    window.lidar_edges_.push_back({1,0,{}});window.refresh();
    if(persistent_map) {
      window.nodes_[0].pose_covariance=M6::Identity()*1e-4;
      window.nodes_[0].covariance_rotation=window.nodes_[0].state.rotation;
      window.insertMap(window.nodes_[0]);window.refresh();
      if(window.lidar_edges_[0].target!=Window::map_target_) throw std::runtime_error("Fixed-map factor not exercised");
    }
    if(structure && (!window.structure_initialized_ || window.nodes_.back().structure.empty()))
      throw std::runtime_error("Synthetic building directions were not observed");
    if(window.lidar_edges_[0].matches.size()<30) throw std::runtime_error("Insufficient synthetic matches");
    Matrix j;Vector residual;double cost;window.linearize(j,residual,cost);
    if(std::abs(cost-window.trialCost())>1e-8) throw std::runtime_error("Linearization/trial objective mismatch");
    const Vector gradient=j.transpose()*residual;Vector numerical(30);
    constexpr double eps=2e-6;
    for(int i=0;i<30;++i) {
      const int node=i/15,axis=i%15;const State saved=window.nodes_[node].state;
      V15 dx=V15::Zero();dx[axis]=eps;
      window.nodes_[node].state=saved.plus(dx);const double plus=window.trialCost();
      window.nodes_[node].state=saved.plus(-dx);const double minus=window.trialCost();
      window.nodes_[node].state=saved;numerical[i]=(plus-minus)/(2*eps);
    }
    const double error=(gradient-numerical).norm()/std::max(1.,numerical.norm());
    std::cout<<"Raw-return LiDAR objective gradient relative error: "<<error<<'\n';
    if(error>2e-5) throw std::runtime_error("Deskew/plane derivatives disagree with actual residual");
    if((!persistent_map && j.block(0,6,j.rows(),9).norm()<1e-3) || j.block(0,21,j.rows(),9).norm()<1e-3)
      throw std::runtime_error("Source or target velocity/bias sensitivity missing");
    // The accepted step must reduce this same raw-return objective, with
    // correspondence identities/plane membership fixed for the trial.
    const State saved=window.nodes_[1].state;V15 step=V15::Zero();step[8]=.2;
    window.nodes_[1].state=saved.plus(step);
    if(std::abs(window.trialCost()-cost)<1e-6) throw std::runtime_error("Trial ignores re-deskewed returns");
  }
};
}
int main() {
  try {temporal_lio::DeskewDerivativeTest::pointJacobians();temporal_lio::DeskewDerivativeTest::structuralGates();
       temporal_lio::DeskewDerivativeTest::run();temporal_lio::DeskewDerivativeTest::run(true);
       temporal_lio::DeskewDerivativeTest::run(false,true);}
  catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}

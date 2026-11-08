#include "temporal_lio.h"
#include <iostream>
#include <stdexcept>
#include <set>

namespace temporal_lio {
struct MapWindowTest {
  static void covariance() {
    State s;s.rotation=expSO3(V3(.3,-.4,.7));s.position=V3(3,-2,1);
    Plane body;body.center=V3(2,-1,3);body.normal=V3(1,2,3).normalized();
    body.additional_covariance.topLeftCorner<3,3>()=.001*(M3::Identity()-body.normal*body.normal.transpose());
    const M6 root=M6::Random(),p=.0001*root*root.transpose();
    const Plane plane=worldPlane(body,s,p),measurement=worldPlane(body,s,M6::Zero());
    M6 j;constexpr double epsilon=1e-6;
    for(int k=0;k<6;++k) {
      V15 dx=V15::Zero();dx[k]=epsilon;
      const auto plus=worldPlane(body,s.plus(dx),M6::Zero()),minus=worldPlane(body,s.plus(-dx),M6::Zero());
      j.col(k).head<3>()=(plus.normal-minus.normal)/(2*epsilon);
      j.col(k).tail<3>()=(plus.center-minus.center)/(2*epsilon);
    }
    if((plane.additional_covariance-measurement.additional_covariance-j*p*j.transpose()).norm()>1e-10)
      throw std::runtime_error("Map-plane covariance disagrees with numerical pose derivative");
    const M3 q=expSO3(V3(-.8,.2,1.3));const V3 shift(10,3,-2),query(4,-1,2);
    State moved=s;moved.rotation=q*s.rotation;moved.position=q*s.position+shift;
    M6 basis=M6::Identity();basis.bottomRightCorner<3,3>()=q;
    const auto transformed=worldPlane(body,moved,basis*p*basis.transpose());
    if(std::abs(planeAdditionalVariance(plane,query)-planeAdditionalVariance(transformed,q*query+shift))>1e-10)
      throw std::runtime_error("Map-plane variance depends on world-frame orientation");
  }
  static void causality() {
    Config c;c.consistent_deskew=true;c.persistent_map=true;c.plane_uncertainty=true;
    c.window_sec=2.;c.max_nodes=6;c.map_max_planes=200;c.max_source_points=500;c.iterations=2;
    Window w(c);State state;M15 covariance=M15::Identity()*1e-4;
    std::vector<ImuSample> imu;
    for(int k=0;k<=820;++k) imu.push_back({.005*k,V3::Zero(),V3(0,0,9.81)});
    std::map<VoxelKey,Plane> seed;
    for(int scan=0;scan<40;++scan) {
      const double time=.1*(scan+1);std::vector<Return> points;
      for(int axis=0;axis<3;++axis) for(int i=0;i<20;++i) for(int j=0;j<20;++j) {
        V3 p=V3::Zero();p[axis]=axis==2 ? -1. : 2.+axis;
        p[(axis+1)%3]=.1+.08*i;p[(axis+2)%3]=.1+.08*j;
        p[axis]+=.003*std::sin(i+j+.6*scan);
        points.push_back({p,time-.095+.004*((i+j)%20),0});
      }
      const auto report=w.add(time,points,imu,state,covariance);
      if(scan==0) seed=w.map_planes_;
      if(scan>0 && report.map_matches<100) throw std::runtime_error("Map did not provide scan observations");
      if(scan<20) {
        if(w.map_planes_.size()!=seed.size()) throw std::runtime_error("Active scan was inserted into map");
        for(const auto& entry:seed) {
          const auto& p=w.map_planes_.at(entry.first);
          if(p.count!=entry.second.count || p.center!=entry.second.center)
            throw std::runtime_error("Map changed before keyframe finalization");
        }
      }
      if(w.map_planes_.size()>static_cast<size_t>(c.map_max_planes) || w.state().position.norm()>.03)
        throw std::runtime_error("Static map drift or unbounded map");
      std::set<uint64_t> sources;
      for(const auto& edge:w.lidar_edges_) if(!sources.insert(edge.source).second)
        throw std::runtime_error("Source scan used in multiple LiDAR factors");
    }
    std::cout<<"Map covariance, causal insertion, single-use scan factors and static replay OK\n";
  }
};
}
int main() {
  try {temporal_lio::MapWindowTest::covariance();temporal_lio::MapWindowTest::causality();}
  catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}

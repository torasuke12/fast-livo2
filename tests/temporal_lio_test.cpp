#include "temporal_lio.h"
#include "temporal_trajectory.h"
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace temporal_lio;
static void require(bool ok,const char* message) {if(!ok) throw std::runtime_error(message);}
static void trajectoryTest() {
  TrajectoryHistory history;
  const auto node = [](uint64_t id, bool keyframe, double height) {
    Node n; n.id=id; n.time=id*.1; n.keyframe=keyframe;
    n.state.position=V3(id*.1,0,height);
    n.state.rotation=expSO3(V3(0,0,id*.01));
    return n;
  };
  history.update({node(1,true,0)});
  history.update({node(1,true,.1),node(2,false,.1)});
  history.update({node(1,true,.2),node(3,false,.2)});
  require(history.poses().size()==2 && history.poses().back().id==3,
          "superseded non-keyframe remains in trajectory");
  history.update({node(1,true,.3),node(4,true,.3)});
  require(history.poses().size()==2 && history.poses().back().id==4,
          "keyframe promotion duplicates live endpoint");
  history.update({node(1,true,-3.45),node(4,true,-3.45),node(5,false,-3.45)});
  require(history.poses().size()==3,"revised trajectory contains stale intermediate poses");
  for (const auto& pose:history.poses())
    require(std::abs(pose.position.z()+3.45)<1e-12,"keyframe correction was not applied");
  history.update({node(4,true,-3.4),node(6,true,-3.4)});
  history.update({node(6,true,-3.3),node(7,false,-3.3)});
  const auto& poses=history.poses();
  require(poses.size()==4 && poses[0].id==1 && poses[1].id==4 && poses[2].id==6 && poses[3].id==7,
          "marginalization loses, duplicates, or reorders trajectory");
  require(std::abs(poses[0].position.z()+3.45)<1e-12 && std::abs(poses[1].position.z()+3.4)<1e-12,
          "marginalized keyframe history was changed");
  require((poses.back().rotation-expSO3(V3(0,0,.07))).norm()<1e-12 &&
          std::abs(poses.back().time-.7)<1e-12,"trajectory loses orientation or timestamp");
  history.update({});
  require(history.poses().size()==4,"empty window damages trajectory history");
}
static void marginalTest() {
  Matrix a=Matrix::Random(45,30),h=a.transpose()*a+Matrix::Identity(30,30);
  Vector b=Vector::Random(30),g;Matrix reduced;
  require(schurComplement(h,b,7,15,reduced,g),"Schur failed");
  Vector all=h.llt().solve(-b),keep=reduced.llt().solve(-g);
  Matrix cov=h.llt().solve(Matrix::Identity(30,30)),small=reduced.llt().solve(Matrix::Identity(15,15));
  int k=0;for(int i=0;i<30;++i) if(i<7 || i>=22) {
    require(std::abs(all[i]-keep[k])<1e-10,"Schur changes retained optimum");
    int l=0;for(int j=0;j<30;++j) if(j<7 || j>=22) {
      require(std::abs(cov(i,j)-small(k,l++))<1e-10,"Schur changes marginal covariance");
    } ++k;
  }
}
static std::vector<ImuSample> stationary(double a,double b,const V3& bg=V3::Zero(),const V3& ba=V3::Zero()) {
  std::vector<ImuSample> out;
  for(int k=0;k<=static_cast<int>(std::round((b-a)/0.005));++k)
    out.push_back({a+k*0.005,bg,V3(0,0,9.81)+ba});
  return out;
}
static void imuTest() {
  Config c;State a;a.gyro_bias=V3(.02,-.01,.03);a.accel_bias=V3(.1,.2,-.1);
  auto samples=stationary(0,.1,a.gyro_bias,a.accel_bias);
  auto d=preintegrate(samples,a,c);const State b=predict(a,d,c.gravity);
  require(b.position.norm()<1e-12 && b.velocity.norm()<1e-12,"stationary IMU drifts");
  require(imuResidual(a,b,d,c.gravity).norm()<1e-12,"IMU factor inconsistent with prediction");
  require(Eigen::SelfAdjointEigenSolver<M15>(d.covariance).eigenvalues().minCoeff()>0,"IMU covariance is not SPD");
  samples.erase(samples.begin()+3,samples.begin()+10);
  bool failed=false;try{interval(samples,0,.1,c.max_imu_gap);}catch(const std::runtime_error&){failed=true;}
  require(failed,"missing IMU silently accepted");
  samples=stationary(0,.1);samples[4].time=samples[3].time;
  failed=false;try{interval(samples,0,.1,c.max_imu_gap);}catch(const std::runtime_error&){failed=true;}
  require(failed,"duplicate IMU silently accepted");
}
static void imuJacobianTest() {
  Config c;State start;start.rotation=expSO3(V3(.4,-.2,.3));
  start.position=V3(2.,-1.,.3);start.velocity=V3(1.2,.5,-.2);
  start.gyro_bias=V3(.02,-.03,.01);start.accel_bias=V3(.1,-.2,.3);
  auto samples=stationary(0,.3);
  for(auto& sample:samples) {sample.gyro=V3(.3,-.1,.2);sample.accel+=V3(.8,-.4,.2);}
  const auto delta=preintegrate(samples,start,c);
  V15 perturb;perturb<<.2,-.1,.15,.3,-.2,.1,.4,-.3,.2,.015,-.02,.01,.08,-.04,.07;
  const State a=start.plus(perturb),b=predict(start,delta,c.gravity).plus(-.4*perturb);
  const auto j=imuJacobians(a,b,delta,c.gravity);
  Eigen::Matrix<double,15,30> numeric;constexpr double epsilon=1e-6;
  for(int k=0;k<15;++k) {
    V15 dx=V15::Zero();dx[k]=epsilon;
    numeric.col(k)=(imuResidual(a.plus(dx),b,delta,c.gravity)-imuResidual(a.plus(-dx),b,delta,c.gravity))/(2*epsilon);
    numeric.col(15+k)=(imuResidual(a,b.plus(dx),delta,c.gravity)-imuResidual(a,b.plus(-dx),delta,c.gravity))/(2*epsilon);
  }
  require((j-numeric).cwiseAbs().maxCoeff()<2e-7,"analytic IMU derivatives disagree at nonzero residual/bias");
  require((j.middleCols<3>(3)+j.middleCols<3>(18)).norm()==0,
          "IMU factor observes common global translation");
  State shifted_a=a,shifted_b=b;const V3 origin(1e6,-2e6,3e6);
  shifted_a.position+=origin;shifted_b.position+=origin;
  const auto shifted=imuJacobians(shifted_a,shifted_b,delta,c.gravity);
  require((shifted.middleCols<3>(3)-j.middleCols<3>(3)).norm()==0 &&
          (shifted.middleCols<3>(18)-j.middleCols<3>(18)).norm()==0,
          "IMU translation Jacobian depends on world origin");
}
static void saturationTest() {
  Config nominal,robust;robust.accel_saturation=3.9*9.81;
  auto samples=stationary(0,.1);
  samples[8].accel.x()=4*9.81;samples[9].accel.x()=-4*9.81;
  State state;
  const auto a=preintegrate(samples,state,nominal),b=preintegrate(samples,state,robust);
  require(b.saturated_intervals==3,"clipping hidden by midpoint averaging");
  const M15 extra=b.covariance-a.covariance;
  require(extra(6,6)>.1 && std::abs(extra(7,7))<1e-10 && std::abs(extra(8,8))<1e-10,
          "clipped acceleration did not inflate the affected axis only");
  require(Eigen::SelfAdjointEigenSolver<M15>(extra).eigenvalues().minCoeff()>-1e-10,
          "saturation handling reduced uncertainty");
  require((a.position-b.position).norm()<1e-12 && (a.velocity-b.velocity).norm()<1e-12 &&
          (a.rotation-b.rotation).norm()<1e-12,"saturation handling invented missing accelerations");
  const auto boundary=interval(samples,.041,.044,.02);
  require(boundary[0].accel.cwiseAbs().maxCoeff()<robust.accel_saturation &&
          boundary[1].accel.cwiseAbs().maxCoeff()<robust.accel_saturation &&
          preintegrate(boundary,state,robust).saturated_intervals==1,
          "boundary interpolation hid a saturated source measurement");
  const auto normal=stationary(0,.1);
  require((preintegrate(normal,state,nominal).covariance-preintegrate(normal,state,robust).covariance).norm()==0,
          "unsaturated IMU covariance changed");
}
static void clippedMotionTest() {
  Config c;c.window_sec=.6;c.keyframe_sec=.1;c.max_nodes=8;c.max_source_points=300;c.iterations=2;
  c.consistent_deskew=true;c.gyro_variance=V3::Constant(.0001);
  Config guarded=c;guarded.accel_saturation=3.9*9.81;
  Window nominal(c),robust(guarded);
  std::vector<ImuSample> measured;std::vector<V3> positions,velocities;
  V3 position=V3::Zero(),velocity=V3::Zero(),previous=V3::Zero();
  // Physical up/down impulses have zero net velocity. Only the positive
  // specific-force peak exceeds the simulated accelerometer's 4 g range.
  for(int k=0;k<=600;++k) {
    const int phase=k%100;V3 acceleration=V3::Zero();
    if(k>=100 && phase>=10 && phase<=11) acceleration.z()=60.;
    if(k>=100 && phase>=12 && phase<=15) acceleration.z()=-30.;
    if(k) {const V3 average=.5*(previous+acceleration);position+=velocity*.005+.5*average*.005*.005;velocity+=average*.005;}
    previous=acceleration;positions.push_back(position);velocities.push_back(velocity);
    const V3 specific=acceleration-c.gravity;
    measured.push_back({k*.005,V3::Zero(),specific.cwiseMax(-4*9.81).cwiseMin(4*9.81)});
  }
  std::vector<V3> scene;
  for(int i=0;i<15;++i) for(int j=0;j<15;++j) {
    const double a=-1.4+.2*i,b=-1.4+.2*j;
    scene.push_back(V3(a,b,-1));scene.push_back(V3(2,a,b));scene.push_back(V3(a,2,b));
  }
  M15 covariance=M15::Identity()*1e-4;covariance.block<3,3>(6,6)*=10;
  covariance.bottomRightCorner<6,6>()*=.01;State initial;
  double normal_error=0,guarded_error=0;
  for(int scan=1;scan<=30;++scan) {
    const double end=scan*.1;std::vector<Return> raw;
    for(size_t i=0;i<scene.size();++i) {
      const double t=end-.095+.09*(i+1)/scene.size();
      const int k=static_cast<int>(t/.005);const double u=t/.005-k;
      const V3 p=(1-u)*positions[k]+u*positions[k+1];raw.push_back({scene[i]-p,t,0});
    }
    nominal.add(end,raw,measured,initial,covariance);robust.add(end,raw,measured,initial,covariance);
    normal_error+=(nominal.state().position-positions[20*scan]).squaredNorm();
    guarded_error+=(robust.state().position-positions[20*scan]).squaredNorm();
  }
  normal_error=std::sqrt(normal_error/30);guarded_error=std::sqrt(guarded_error/30);
  std::cout<<"clipped impulses: position RMS nominal="<<normal_error<<" guarded="<<guarded_error<<'\n';
  require(guarded_error<.03 && guarded_error<.5*normal_error,
          "LiDAR could not correct motion corrupted by clipped acceleration");
}
static void deskewTest() {
  Config c;c.lidar_to_imu_rotation=expSO3(V3(.2,-.1,.3));c.lidar_to_imu_translation=V3(.1,-.2,.04);
  State end;end.rotation=expSO3(V3(0,0,.1));end.position=V3(.2,0,0);end.velocity=V3(2,0,0);
  auto imu=stationary(0,.1);for(auto& q:imu) q.gyro=V3(0,0,1);
  V3 world(3,1,.3);std::vector<Return> raw;
  for(int k=0;k<=100;++k) {
    double t=k*.001;M3 r=expSO3(V3(0,0,t));V3 p(2*t,0,0);
    raw.push_back({c.lidar_to_imu_rotation.transpose()*(r.transpose()*(world-p)-c.lidar_to_imu_translation),t,0});
  }
  auto points=deskew(raw,imu,.1,end,c);
  for(const auto& q:points) require((end.rotation*q+end.position-world).norm()<1e-9,"deskew timing/frame/extrinsic error");
}
static void initializationReferenceTest() {
  Config c;c.keyframe_sec=.1;c.window_sec=.25;c.max_nodes=5;c.iterations=1;
  std::vector<V3> seed;
  for(int i=0;i<20;++i) for(int j=0;j<20;++j)
    seed.push_back(V3(1.01+i*.04,2.01+j*.04,-1));
  State s;const M15 covariance=M15::Identity()*.01;
  Window seeded(c),single(c);
  const std::vector<Return> sparse{{V3(5,6,7),.05,0}};
  const auto first=seeded.add(.1,sparse,stationary(0,.11),s,covariance,seed);
  single.add(.1,sparse,stationary(0,.11),s,covariance);
  require(first.status=="initialized" && first.matches==0 && first.imu_edges==0 &&
          (first.covariance-covariance).norm()<1e-12,"bootstrap added a self measurement or changed the prior");
  require(single.nodes().front().planes.empty() && !seeded.nodes().front().planes.empty(),
          "accumulated seed was not used for the first reference planes");
  require(seeded.nodes().front().raw.size()==1 && seeded.nodes().front().points.size()==1 &&
          (seeded.nodes().front().points.front()-sparse.front().point).norm()<1e-12,
          "bootstrap geometry replaced/re-deskewed current raw returns");
  bool failed=false;
  try {seeded.add(.2,sparse,stationary(.09,.21),s,covariance,seed);}
  catch(const std::invalid_argument&) {failed=true;}
  require(failed && seeded.nodes().size()==1,"late bootstrap injection was accepted or mutated the window");
  std::vector<Return> dense;
  for(const auto& p:seed) dense.push_back({p,.15,0});
  const auto next=seeded.add(.2,dense,stationary(.09,.21),s,covariance);
  require(next.matches>10 && seeded.state().position.norm()<1e-8,"next scan did not match the initial accumulated reference");
  require(seeded.nodes().back().initialization_reference.empty(),"bootstrap leaked into later keyframes");
  require(seeded.nodes().front().initialization_reference==seed,"refresh changed frozen bootstrap geometry");
  for(int k=3;k<=5;++k) {
    for(auto& p:dense) p.time=k*.1-.05;
    seeded.add(k*.1,dense,stationary(k*.1-.11,k*.1+.01),s,covariance);
  }
  for(const auto& n:seeded.nodes()) require(n.initialization_reference.empty(),"bootstrap survived removal of the initial keyframe");
}
static void boundedWindowTest() {
  Config full,lag;full.window_sec=10;full.keyframe_sec=.1;full.max_nodes=32;full.iterations=1;
  lag=full;lag.window_sec=.8;lag.keyframe_sec=.3;lag.max_nodes=5;
  Window batch(full),window(lag);State s;M15 cov=M15::Identity()*.01;Report a,b;
  for(int k=1;k<=20;++k) {
    double t=k*.1;std::vector<Return> raw{{V3(2,0,0),t-.09,0}};auto imu=stationary(0,t+.01);
    a=batch.add(t,raw,imu,s,cov);b=window.add(t,raw,imu,s,cov);
    require(window.nodes().size()<=5 && b.span<=.8+1e-6,"window not bounded");
    require((a.covariance-b.covariance).norm()/a.covariance.norm()<2e-5,"marginalization loses/double-counts IMU information");
    require(window.state().velocity.norm()<1e-6,"unobserved stationary state drift");
  }
  require(b.covariance(3,3)>cov(3,3),"unobserved translation covariance falsely shrinks");
  const double start_variance=b.covariance(3,3);
  for(int k=21;k<=1000;++k) {
    const double t=k*.1;
    b=window.add(t,{{V3(2,0,0),t-.09,0}},stationary(t-.11,t+.01),s,cov);
    require(b.status=="imu_only" && window.nodes().size()<=5,"long unobserved window changed mode or size");
  }
  require(b.covariance(3,3)>start_variance*100,"long unobserved trajectory acquired spurious position information");
  Config shorter=lag;shorter.window_sec=.05;shorter.keyframe_sec=.05;Window tight(shorter);
  for(int k=1;k<=3;++k) {
    const double t=k*.1;
    const auto report=tight.add(t,{{V3(2,0,0),t-.09,0}},stationary(0,t+.01),s,cov);
    require(report.span<=shorter.window_sec,"scan interval larger than window breaks horizon bound");
  }
}
static State truth(double t) {
  State s;
  s.rotation=expSO3(V3(0,0,.7*t))*expSO3(V3(.12*std::sin(1.7*t),.15*std::sin(1.1*t),0));
  s.position=V3(.6*std::sin(.6*t),.5*(1-std::cos(.6*t)),.1*std::sin(1.2*t));
  s.velocity=V3(.36*std::cos(.6*t),.3*std::sin(.6*t),.12*std::cos(1.2*t));
  s.gyro_bias=V3(.012,-.008,.006);s.accel_bias=V3(.12,-.08,.06);return s;
}
static void corridorTest() {
  Config c;c.window_sec=1.0;c.keyframe_sec=.2;c.max_nodes=8;c.max_source_points=300;c.iterations=2;
  Window w(c);State initial;M15 covariance=M15::Identity()*.01;
  covariance.topLeftCorner<6,6>()*=.01;
  Report report;
  for(int k=1;k<=20;++k) {
    const double t=k*.1;std::vector<Return> raw;
    for(int i=-20;i<=20;++i) for(int j=-10;j<=10;++j) {
      raw.push_back({V3(i*.1,j*.1,-1),t-.05,0});
      raw.push_back({V3(i*.1,2,j*.1),t-.05,0});
    }
    report=w.add(t,raw,stationary(t-.11,t+.01,V3::Zero(),V3(.1,0,0)),initial,covariance);
  }
  require(report.matches>100,"corridor did not provide plane observations");
  require(std::abs(w.state().accel_bias.x())<1e-4,"unobservable forward bias acquired spurious information");
  require(report.covariance(3,3)>report.covariance(4,4)*5,"corridor forward uncertainty falsely constrained");
  require(w.state().velocity.x()>.15,"corridor silently imposed a forward velocity observation");
}
static void jointTest() {
  Config c;c.window_sec=2.4;c.keyframe_sec=.3;c.max_nodes=12;c.iterations=3;c.max_source_points=450;c.sample_size=.12;
  c.gyro_variance=V3::Constant(.002);c.accel_variance=V3::Constant(.02);
  Window w(c);State initial=truth(.1);initial.gyro_bias.setZero();initial.accel_bias.setZero();
  M15 covariance=M15::Identity()*.1;covariance.topLeftCorner<6,6>()*=.0001;
  const M3 change=expSO3(V3(.4,-.3,.6));const V3 translation(12,-7,3);
  Config rotated=c;rotated.gravity=change*c.gravity;Window other(rotated);
  State initial_other=initial;initial_other.rotation=change*initial.rotation;
  initial_other.position=change*initial.position+translation;initial_other.velocity=change*initial.velocity;
  M15 transport=M15::Identity();transport.block<3,3>(3,3)=change;transport.block<3,3>(6,6)=change;
  const M15 covariance_other=transport*covariance*transport.transpose();
  std::vector<V3> scene;
  for(int i=-18;i<=18;++i) for(int j=-18;j<=18;++j) {
    const double a=i*.1,b=j*.1;scene.push_back(V3(a,b,-1));scene.push_back(V3(2,a,b));scene.push_back(V3(a,2,b));
  }
  std::vector<ImuSample> imu;
  for(int k=0;k<=1002;++k) {
    const double t=k*.005,eps=1e-5;State a=truth(t);
    const V3 gyro=logSO3(truth(t-eps).rotation.transpose()*truth(t+eps).rotation)/(2*eps)+a.gyro_bias;
    const V3 aw(-.216*std::sin(.6*t),.18*std::cos(.6*t),-.144*std::sin(1.2*t));
    imu.push_back({t,gyro,a.rotation.transpose()*(aw-c.gravity)+a.accel_bias});
  }
  Report report,report_other;
  for(int k=1;k<=50;++k) {
    double end=k*.1;std::vector<Return> raw;
    for(size_t i=0;i<scene.size();++i) {
      double t=end-.1+.1*(i+1)/scene.size();auto s=truth(t);
      raw.push_back({s.rotation.transpose()*(scene[i]-s.position),t,0});
    }
    report=w.add(end,raw,imu,initial,covariance);
    report_other=other.add(end,raw,imu,initial_other,covariance_other);
  }
  const auto s=w.state(),expected=truth(5);
  std::cout<<"joint: gyro bias error="<<(s.gyro_bias-expected.gyro_bias).norm()
    <<" accel bias error="<<(s.accel_bias-expected.accel_bias).norm()
    <<" velocity error="<<(s.velocity-expected.velocity).norm()<<" matches="<<report.matches
    <<" ms="<<report.solve_ms<<'\n';
  require((s.gyro_bias-expected.gyro_bias).norm()<.006,"joint gyro bias did not recover");
  require((s.accel_bias-expected.accel_bias).norm()<.075,"joint accel bias did not recover");
  require((s.velocity-expected.velocity).norm()<.1,"joint velocity inconsistent");
  require(report.matches>150,"synthetic geometry not matched");
  require((other.state().position-change*s.position-translation).norm()<1e-4,"world-frame choice changes optimized position");
  require((other.state().accel_bias-s.accel_bias).norm()<1e-4,"world-frame choice changes body bias");
  require((report_other.covariance-transport*report.covariance*transport.transpose()).norm()<1e-4,
          "joint covariance uses inconsistent coordinate frames");
}
int main() {
  try {trajectoryTest();marginalTest();imuTest();imuJacobianTest();saturationTest();clippedMotionTest();deskewTest();initializationReferenceTest();boundedWindowTest();corridorTest();jointTest();}
  catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
  std::cout<<"temporal LIO tests passed\n";return 0;
}

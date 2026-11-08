#include "temporal_lio.h"
#include <Eigen/Cholesky>
#include <iostream>
#include <stdexcept>

namespace temporal_lio {
struct LandmarkWindowTest {
  static std::vector<Return> scan(double time,int number=0) {
    std::vector<Return> points;
    for(int axis=0;axis<3;++axis) for(int i=0;i<40;++i) for(int j=0;j<32;++j) {
      V3 p=V3::Zero();p[axis]=axis==2 ? -1. : 2.+axis;
      p[(axis+1)%3]=-.9+.08*i;p[(axis+2)%3]=-.9+.08*j;
      p[axis]+=.001*std::sin(i+j+.6*number);
      points.push_back({p,time-.095+.004*((i+j)%20),0});
    }
    // A ramp is deliberately not a Manhattan surface. Rejecting this plane
    // would remove its vertical information during a stair/ramp traversal.
    const V3 normal=V3(0,.5,1).normalized();const auto tangent=planeBasis(normal);
    for(int i=0;i<25;++i) for(int j=0;j<25;++j) {
      const V3 p=V3(-3,0,-2)+tangent.col(0)*(-1.+.1*i)+tangent.col(1)*(-1.+.1*j);
      points.push_back({p,time-.095+.004*((i+j)%20),0});
    }
    return points;
  }
  static Config config() {
    Config c;c.consistent_deskew=true;c.structural_landmarks=true;c.structure_enabled=true;
    c.window_sec=1.;c.max_nodes=4;c.max_landmarks=32;c.max_source_points=300;c.iterations=2;
    c.structure_min_planes=3;return c;
  }
  static std::vector<ImuSample> imu() {
    std::vector<ImuSample> samples;
    for(int k=0;k<=820;++k) samples.push_back({.005*k,V3::Zero(),V3(0,0,9.81)});
    return samples;
  }
  static void uncertainDirection() {
    auto c=config();c.plane_uncertainty=true;
    Window guarded(c);guarded.structure_initialized_=true;guarded.structure_axes_=M3::Identity();
    Node node;node.state=State{};
    // A narrow, perfectly planar patch can have an uncertain normal even
    // though its smallest scatter eigenvalue and its axis-angle are both zero.
    // This reproduces the six-point, weak corridor patch seen in the raw bag.
    for(double y:{.20,.25}) for(double z:{.10,.22,.34})
      node.points.push_back(V3(4.2,y,z));
    node.planes=fitPlanes(node.points,c);
    if(node.planes.size()!=1) throw std::runtime_error("Uncertain patch fixture was not planar");
    for(size_t i=0;i<node.points.size();++i) node.source_indices.push_back(i);
    guarded.nodes_.push_back(node);guarded.prior_origins_.push_back(node.state);
    guarded.prior_jacobian_=M15::Identity();guarded.prior_residual_=V15::Zero();
    Window unguarded=guarded;unguarded.config_.plane_uncertainty=false;
    unguarded.refreshLandmarks();guarded.refreshLandmarks();
    if(unguarded.landmarks_.at(0).structural_axis!=0 || guarded.landmarks_.at(0).structural_axis!=-1 ||
       guarded.prior_jacobian_.rows()!=15 || guarded.nodes_.back().landmark_matches.size()!=6)
      throw std::runtime_error("Weak normal invented a structural direction or its raw points were discarded");
    // Weak source support may still observe a direction established by other,
    // reliable returns. It must neither add another prior nor duplicate points.
    guarded=unguarded;guarded.config_.plane_uncertainty=true;
    guarded.nodes_.back().landmarks_assigned=false;guarded.nodes_.back().landmark_matches.clear();
    const int rows=guarded.prior_jacobian_.rows();guarded.refreshLandmarks();
    if(guarded.landmarks_.size()!=1 || guarded.prior_jacobian_.rows()!=rows ||
       guarded.nodes_.back().landmark_matches.size()!=6)
      throw std::runtime_error("Weak patch lost an established plane or duplicated its direction prior");
    // Quality ordering must be deterministic and source slots must continue
    // to address the original raw-return Jacobians after reordering.
    Window stable(c);const auto samples=imu();const M15 cov=M15::Identity()*.0001;
    for(int k=0;k<12;++k) {
      const double time=.1*(k+1);stable.add(time,scan(time,k),samples,State{},cov);
      if(stable.state().position.norm()>.015) throw std::runtime_error("Quality-aware static plane graph drifted");
      for(const auto& n:stable.nodes_) for(const auto& m:n.landmark_matches)
        if(n.source_indices.at(m.slot)!=m.source) throw std::runtime_error("Quality ordering changed raw Jacobian slots");
    }
  }
  static void run() {
    // Exactly adjacent local supports must associate identically after a
    // world-origin translation. The next micrometre outside still separates.
    const auto support_graph=[](const V3& origin,double gap) {
      Window w(config());w.structure_initialized_=true;w.structure_axes_=M3::Identity();
      Node node;node.state.position=origin;
      for(double center_x:{-.7600000000000001,1.24+gap}) {
        Plane plane;plane.center=V3(center_x,.2,-1);plane.normal=V3::UnitZ();plane.count=6;
        w.config_.plane_uncertainty=false;
        node.planes.emplace(voxelKey(plane.center,w.config_.voxel_size),plane);
        for(double x:{-.01,.01}) for(double y:{-.02,0.,.02}) {
          node.source_indices.push_back(node.points.size());
          node.points.push_back(plane.center+V3(x,y,0));
        }
      }
      w.nodes_.push_back(node);w.prior_origins_={node.state};
      w.prior_jacobian_=M15::Identity();w.prior_residual_=V15::Zero();w.refreshLandmarks();
      return w;
    };
    const auto local=support_graph(V3::Zero(),0),translated=support_graph(V3(100,30,-2),0);
    if(local.landmarks_.size()!=1 || translated.landmarks_.size()!=1 ||
       support_graph(V3::Zero(),1e-6).landmarks_.size()!=2 ||
       support_graph(V3(100,30,-2),1e-6).landmarks_.size()!=2)
      throw std::runtime_error("Plane support boundary depends on world origin or accepts outside support");
    uncertainDirection();
    // Global position is not observed by raw point-to-plane/IMU factors.
    // Increasing its independent initial covariance must survive repeated
    // changes of the window anchor without changing relative motion or bias.
    {
      Config c=config();c.plane_uncertainty=true;c.window_sec=.6;c.keyframe_sec=.2;
      c.structure_sigma=.01*3.141592653589793/180.;c.iterations=3;
      Window precise(c),uncertain(c);State initial;initial.velocity=V3(1,.05,0);
      const auto samples=imu();const M15 small=M15::Identity()*1e-4;
      M15 large=small,extra=M15::Zero();extra.block<3,3>(3,3)=M3::Identity();large+=extra;
      for(int k=0;k<24;++k) {
        const double time=.1*(k+1);auto returns=scan(time,k);
        for(auto& p:returns) p.point+=V3(.013,.017,.019)-initial.velocity*p.time;
        const auto a=precise.add(time,returns,samples,initial,small);
        const auto b=uncertain.add(time,returns,samples,initial,large);
        if((b.covariance-a.covariance-extra).norm()>2e-6 ||
           uncertain.state().minus(precise.state()).norm()>1e-5)
          throw std::runtime_error("Plane marginalization invented absolute-position information");
      }
    }
    Window w(config());const auto samples=imu();State initial;
    const M15 covariance=M15::Identity()*.0001;
    for(int k=0;k<25;++k) {
      const double time=.1*(k+1);
      const auto report=w.add(time,scan(time,k),samples,initial,covariance);
      if(w.landmarks_.size()<3 || w.landmarks_.size()>32 || w.state().position.norm()>.015)
        throw std::runtime_error("Joint plane map unbounded or static scene drifted");
      if(!w.lidar_edges_.empty() || !w.nodes_.back().structure.empty())
        throw std::runtime_error("Structural raw returns counted twice");
      if(Eigen::LLT<M15>(report.covariance).info()!=Eigen::Success)
        throw std::runtime_error("Joint plane marginal covariance not positive definite");
      if(k<2) continue;
      Matrix j;Vector r;double cost;w.linearize(j,r,cost);
      if(j.cols()!=static_cast<int>(15*w.nodes_.size()+3*w.landmarks_.size()))
        throw std::runtime_error("Joint plane prior columns lost after marginalization");
      if(std::abs(cost-w.trialCost())>1e-7*std::max(1.,cost))
        throw std::runtime_error("Joint plane cost disagrees with linearization");
    }
    // Independent linear-Gaussian reference for the reported current-state
    // marginal, including correlations with all three plane coordinates.
    Matrix covariance_j;Vector covariance_r;double covariance_cost;
    w.linearize(covariance_j,covariance_r,covariance_cost);
    const Matrix h=covariance_j.transpose()*covariance_j;
    Matrix selector=Matrix::Zero(h.rows(),15);
    selector.middleRows(15*(w.nodes_.size()-1),15)=M15::Identity();
    const M15 expected_covariance=selector.transpose()*h.ldlt().solve(selector);
    Report marginal;w.marginalCovariance(marginal);
    if((expected_covariance-marginal.covariance).norm()>1e-6*expected_covariance.norm())
      throw std::runtime_error("Reported state marginal lost plane correlations");
    bool ramp_observed=false;
    for(const auto& landmark:w.landmarks_)
      if(std::abs(landmark.normal.dot(V3(0,.5,1).normalized()))>.999) ramp_observed=true;
    if(!ramp_observed) throw std::runtime_error("Inclined planes were discarded by structural classification");
    // A free plane may approach a structural axis during optimization. Its
    // original lack of a direction prior must not be inherited by floor points.
    Window classified=w;
    for(auto& landmark:classified.landmarks_) if(landmark.structural_axis<0) {
      landmark.normal=V3::UnitZ();landmark.distance=-1.;
    }
    auto& classified_node=classified.nodes_.back();
    classified_node.landmark_matches.clear();classified_node.landmarks_assigned=false;
    classified.refreshLandmarks();
    for(const auto& match:classified_node.landmark_matches) {
      const auto& plane=classified_node.planes.at(voxelKey(classified_node.points[match.source],classified.config_.voxel_size));
      const V3 dots=classified.structure_axes_.transpose()*classified_node.state.rotation*plane.normal;
      if(dots.cwiseAbs().maxCoeff()>std::cos(classified.config_.structure_max_angle) &&
         classified.landmarks_[match.landmark].structural_axis<0)
        throw std::runtime_error("Free plane absorbed a structural observation");
    }
    // Perturb both pose/bias and shared plane distance, including coordinates
    // that are correlated in an existing marginal prior.
    V15 perturb=V15::Zero();perturb<<.001,-.002,.001,.003,-.004,.002,.01,.02,-.01,.001,.002,-.001,.01,-.02,.01;
    for(auto& n:w.nodes_) n.state=n.state.plus(perturb);
    for(auto& l:w.landmarks_) l.distance+=.003;
    w.refresh();Matrix j;Vector r;double cost;w.linearize(j,r,cost);
    const Vector gradient=j.transpose()*r;
    double error=0;constexpr double eps=1e-6;
    for(int k=0;k<j.cols();++k) {
      double plus,minus;
      if(k<static_cast<int>(15*w.nodes_.size())) {
        auto& s=w.nodes_[k/15].state;const State saved=s;
        V15 dx=V15::Zero();dx[k%15]=eps;
        s=saved.plus(dx);plus=w.trialCost();s=saved.plus(-dx);minus=w.trialCost();s=saved;
      } else {
        const int coordinate=k-15*w.nodes_.size();auto& landmark=w.landmarks_[coordinate/3];const auto saved=landmark;
        if(coordinate%3==2) {
          landmark.distance=saved.distance+eps;plus=w.trialCost();landmark.distance=saved.distance-eps;minus=w.trialCost();
        } else {
          const V3 tangent=planeBasis(saved.normal).col(coordinate%3);
          landmark.normal=(saved.normal+eps*tangent).normalized();plus=w.trialCost();
          landmark.normal=(saved.normal-eps*tangent).normalized();minus=w.trialCost();
        }
        landmark=saved;
      }
      error=std::max(error,std::abs((plus-minus)/(2*eps)-gradient[k])/std::max(1.,std::abs(gradient[k])));
    }
    if(error>2e-5) throw std::runtime_error("Joint plane objective gradient mismatch "+std::to_string(error));
    // Removing a state must preserve exactly the Schur complement of touching
    // factors, including all cross terms with persistent plane distances.
    // A no-longer-observed plane must also be eliminated from the prior.
    const int prior_width=w.prior_jacobian_.cols(),prior_rows=w.prior_jacobian_.rows();
    Matrix expanded=Matrix::Zero(prior_rows+3,prior_width+3);
    expanded.topLeftCorner(prior_rows,prior_width)=w.prior_jacobian_;
    expanded.bottomRightCorner<3,3>()=3.*M3::Identity();w.prior_jacobian_=expanded;
    w.prior_residual_.conservativeResize(prior_rows+3);w.prior_residual_.tail<3>().setConstant(.2);
    w.landmarks_.push_back({V3::UnitZ(),V3::Zero(),V3::Zero(),3.});w.prior_landmark_origins_.push_back(w.landmarks_.back());
    // Re-anchoring is an invertible coordinate change. It must preserve the
    // current marginal and objective before eliminating the previous anchor.
    Report before_anchor,after_anchor;w.marginalCovariance(before_anchor);
    const double old_cost=w.trialCost();w.reanchorPlanes(w.nodes_[1].id);
    w.marginalCovariance(after_anchor);
    if((before_anchor.covariance-after_anchor.covariance).norm()>1e-8*before_anchor.covariance.norm() ||
       std::abs(old_cost-w.trialCost())>1e-8*std::max(1.,old_cost))
      throw std::runtime_error("Plane re-anchoring changed pose information or objective");
    w.linearize(j,r,cost,0);Matrix expected;Vector expected_rhs;
    if(!schurComplement(j.transpose()*j,j.transpose()*r,0,15,expected,expected_rhs))
      throw std::runtime_error("Reference Schur complement failed");
    Matrix retired;Vector retired_rhs;
    if(!schurComplement(expected,expected_rhs,expected.rows()-3,3,retired,retired_rhs))
      throw std::runtime_error("Reference retired-plane Schur complement failed");
    expected=retired;expected_rhs=retired_rhs;
    w.marginalize(0);
    const Matrix actual=w.prior_jacobian_.transpose()*w.prior_jacobian_;
    if((actual-expected).norm()>1e-7*expected.norm() ||
       (w.prior_jacobian_.transpose()*w.prior_residual_-expected_rhs).norm()>1e-7*std::max(1.,expected_rhs.norm()))
      throw std::runtime_error("Plane/state correlations lost during marginalization");
    Window corridor(config());
    Report corridor_report;
    for(int k=0;k<25;++k) {
      const double time=.1*(k+1);auto points=scan(time,k);
      points.erase(points.begin(),points.begin()+40*32); // floor and wall only; no x normal
      corridor_report=corridor.add(time,points,samples,initial,covariance);
    }
    if(corridor_report.covariance(3,3)<10*covariance(3,3) ||
       corridor_report.covariance(3,3)<5*corridor_report.covariance(4,4))
      throw std::runtime_error("Structural corridor invented longitudinal information");
    std::cout<<"Joint plane gradient error "<<error<<"; static scene and marginal prior correlations OK\n";
  }
};
}
int main() {
  try {temporal_lio::LandmarkWindowTest::run();}
  catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}

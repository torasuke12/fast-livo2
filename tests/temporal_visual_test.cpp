#include "temporal_lio.h"
#include <Eigen/Cholesky>
#include <algorithm>
#include <iostream>
#include <stdexcept>
using namespace temporal_lio;
static void require(bool ok,const char* text) {if(!ok) throw std::runtime_error(text);}

int main() {
  try {
    // Compare elimination of per-image exposure against a dense Schur result.
    Matrix j=Matrix::Random(80,7);Vector r=Vector::Random(80);State origin;
    const auto o=photometricObservation(origin,j,r,3.0);
    const Matrix h=j.transpose()*j/3.0;const Vector g=j.transpose()*r/3.0;
    const Matrix expected_h=h.topLeftCorner(6,6)-h.topRightCorner(6,1)*h.bottomLeftCorner(1,6)/h(6,6);
    const Vector expected_g=g.head(6)-h.topRightCorner(6,1)*g[6]/h(6,6);
    require((o.jacobian.transpose()*o.jacobian-expected_h).norm()<1e-10,"exposure elimination changes image information");
    require((o.jacobian.transpose()*o.residual-expected_g).norm()<1e-10,"exposure elimination changes image gradient");
    std::vector<PoseObservation::Reference> anchors{{3,origin,Matrix::Zero(80,6)},{4,origin,Matrix::Zero(80,6)}};
    for(int k=0;k<2;++k) {
      anchors[k].jacobian.block(k*40,0,40,3)=Matrix::Random(40,3);
      anchors[k].jacobian.block(k*40,3,40,3)=-j.block(k*40,3,40,3);
    }
    const auto anchored=photometricObservation(origin,j,r,3.,anchors);
    Matrix joint_j(80,18);joint_j<<j.leftCols(6),anchors[0].jacobian,anchors[1].jacobian;
    Matrix compressed(anchored.residual.size(),18);
    compressed<<anchored.jacobian,anchored.references[0].jacobian,anchored.references[1].jacobian;
    const Matrix projector=Matrix::Identity(80,80)-j.col(6)*j.col(6).transpose()/j.col(6).squaredNorm();
    require((compressed.transpose()*compressed-joint_j.transpose()*projector*joint_j/3.).norm()<1e-9,
            "joint image compression lost reference cross-information");
    require((compressed.transpose()*anchored.residual-joint_j.transpose()*projector*r/3.).norm()<1e-9,
            "joint image compression changed reference gradient");
    Matrix gauge=Matrix::Zero(18,3);
    for(int k=0;k<3;++k) gauge.block<3,3>(6*k+3,0).setIdentity();
    require((compressed*gauge).norm()<1e-10,"relative image invented a global position measurement");
    State anchor;anchor.rotation=expSO3(V3(.2,-.1,.3));anchor.position=V3(1,-2,3);
    const V3 body(.5,1.,4.),world=anchor.rotation*body+anchor.position;
    Matrix world_j(2,3);world_j<<2,-1,.3,-.2,.4,1.;
    const Matrix analytic=pointAnchorJacobian(anchor,world,-world_j);
    Matrix numeric(2,6);
    for(int k=0;k<6;++k) {
      V15 d=V15::Zero();d[k]=1e-6;
      const auto plus=anchor.plus(d),minus=anchor.plus(-d);
      numeric.col(k)=world_j*(plus.rotation*body+plus.position-minus.rotation*body-minus.position)/(2e-6);
    }
    require((analytic-numeric).norm()<1e-8,"point reference derivative is inconsistent with anchor motion");
    M3 geometry=M3::Zero();geometry(1,1)=geometry(2,2)=100;
    const auto weak=weakTranslationObservation(o,geometry,.05);
    const auto anchored_weak=weakTranslationObservation(anchored,geometry,.05);
    Matrix weak_joint(anchored_weak.residual.size(),18);
    weak_joint<<anchored_weak.jacobian,anchored_weak.references[0].jacobian,anchored_weak.references[1].jacobian;
    require((weak_joint*gauge).norm()<1e-9,"weak translation filtering destroyed the image gauge");
    Matrix reorder=Matrix::Zero(6,6);int column=0;
    for(int index:{0,1,2,4,5,3}) reorder(index,column++)=1.;
    const Matrix ordered_h=reorder.transpose()*expected_h*reorder;
    const Vector ordered_g=reorder.transpose()*expected_g;
    const double weak_h=ordered_h(5,5)-(ordered_h.bottomLeftCorner(1,5)*ordered_h.topLeftCorner(5,5).llt().solve(ordered_h.topRightCorner(5,1)))(0,0);
    const double weak_g=ordered_g[5]-(ordered_h.bottomLeftCorner(1,5)*ordered_h.topLeftCorner(5,5).llt().solve(ordered_g.head(5)))[0];
    M6 target_h=M6::Zero();target_h(3,3)=weak_h;Eigen::Matrix<double,6,1> target_g=Eigen::Matrix<double,6,1>::Zero();target_g[3]=weak_g;
    require((weak.jacobian.transpose()*weak.jacobian-target_h).norm()<1e-10 &&
            (weak.jacobian.transpose()*weak.residual-target_g).norm()<1e-10,
            "weak image direction disagrees with marginal Gaussian information");
    auto nuisance=o;nuisance.residual+=10*o.jacobian.col(4)+2*o.jacobian.col(0);
    const auto filtered=weakTranslationObservation(nuisance,geometry,.05);
    require((filtered.jacobian.transpose()*filtered.residual-target_g).norm()<1e-10,
            "image error in a nuisance direction leaked into corridor translation");
    require(weakTranslationObservation(o,M3::Identity(),.05).jacobian.rows()==0,
            "fully constrained geometry received a restricted visual factor");
    const M3 rotation=expSO3(V3(.3,.5,-.4));M6 transform=M6::Identity();transform.bottomRightCorner<3,3>()=rotation;
    auto moved=o;moved.jacobian=o.jacobian*transform.transpose();
    const auto equivariant=weakTranslationObservation(moved,rotation*geometry*rotation.transpose(),.05);
    require((equivariant.jacobian.transpose()*equivariant.jacobian-transform*target_h*transform.transpose()).norm()<1e-10,
            "visual direction selection depends on the world coordinate axes");
    // Brightness alone must not become a pose constraint.
    j.setZero();j.col(6).setOnes();r.setConstant(8);
    const auto brightness=photometricObservation(origin,j,r,1);
    require(brightness.jacobian.norm()==0,"brightness adds pose information");

    Config c;c.window_sec=.21;c.keyframe_sec=.1;c.iterations=3;
    Window w(c),reference(c);M15 prior=M15::Identity()*.1;
    prior(3,6)=prior(6,3)=.03;prior(3,12)=prior(12,3)=.003;
    prior(3,11)=prior(11,3)=.002;
    std::vector<ImuSample> imu;
    for(int k=0;k<=200;++k) imu.push_back({k*.005,V3::Zero(),V3(0,0,9.81)});
    auto raw=[](double t){return std::vector<Return>{{V3(0,0,-1),t-.001,0}};};
    w.add(.1,raw(.1),imu,origin,prior);
    Matrix image_j=Matrix::Zero(1,6);image_j(0,3)=1;Vector image_r=Vector::Constant(1,-.05);
    auto measurement=photometricObservation(origin,image_j,image_r,.01);
    const Report fused=w.addVisual(.1,measurement);
    const V15 increment=prior.col(3)*(.05/(prior(3,3)+.01));
    const M15 posterior=prior-prior.col(3)*prior.row(3)/(prior(3,3)+.01);
    require(fused.status=="updated","image was not fused");
    require((w.state().minus(origin)-increment).norm()<1e-9,"visual mean disagrees with Gaussian update");
    require((fused.covariance-posterior).norm()<1e-9,"visual prior counted twice");
    require(w.state().velocity.x()>0 && w.state().accel_bias.x()>0 && w.state().gyro_bias.z()>0,
            "visual information not transferred to velocity and both biases");
    bool duplicate=false;try {w.addVisual(.1,measurement);} catch(const std::invalid_argument&) {duplicate=true;}
    require(duplicate,"duplicate image accepted");
    bool wrong_time=false;try {w.addVisual(.2,measurement);} catch(const std::invalid_argument&) {wrong_time=true;}
    require(wrong_time,"image at wrong LiDAR time accepted");
    reference.add(.1,raw(.1),imu,origin.plus(increment),posterior);
    // Once its node leaves the lag, the image survives in the prior once.
    for(int k=2;k<=8;++k) {
      double t=k*.1;
      auto a=w.add(t,raw(t),imu,origin,prior);
      auto b=reference.add(t,raw(t),imu,origin,prior);
      require((w.state().minus(reference.state())).norm()<1e-6,"image update lost or repeated after next scan/marginalization");
      require((a.covariance-b.covariance).norm()<1e-6,"visual information lost or counted twice after marginalization");
    }
    const State saved=w.state();measurement.origin=saved;measurement.residual[0]=-100;
    require(w.addVisual(.8,measurement).status=="rejected","catastrophic image update accepted");
    require(w.state().minus(saved).norm()<1e-12,"rejection failed to restore state");

    // With zero innovations this is a linear Gaussian system. Removing a
    // reference pose must transfer its image factor exactly once, including
    // its cross-covariance with the still-retained current pose.
    Config full_config=c;full_config.window_sec=2.;full_config.max_nodes=16;
    Window lag(c),full(full_config);
    lag.add(.1,raw(.1),imu,origin,prior);full.add(.1,raw(.1),imu,origin,prior);
    for(int k=2;k<=8;++k) {
      const double t=k*.1;
      lag.add(t,raw(t),imu,origin,prior);full.add(t,raw(t),imu,origin,prior);
      PoseObservation pair;pair.origin=origin;pair.jacobian=Matrix::Zero(1,6);pair.jacobian(0,3)=10.;
      pair.residual=Vector::Zero(1);
      pair.references.push_back({static_cast<uint64_t>(k-2),origin,-pair.jacobian});
      const auto a=lag.addVisual(t,pair),b=full.addVisual(t,pair);
      require(lag.state().minus(full.state()).norm()<1e-10,"reference image lost or repeated during marginalization");
      require((a.covariance-b.covariance).norm()<1e-9,"reference cross-covariance lost during marginalization");
    }
    // A finite correction relinearizes old IMU factors in the full window,
    // while the lag retains their first-order prior. Check a small response
    // separately from exact information preservation at zero innovation.
    lag.add(.9,raw(.9),imu,origin,prior);full.add(.9,raw(.9),imu,origin,prior);
    PoseObservation small;small.origin=origin;small.jacobian=Matrix::Zero(1,6);small.jacobian(0,3)=10.;
    small.residual=Vector::Constant(1,-1e-5);small.references.push_back({7,origin,-small.jacobian});
    const auto lag_small=lag.addVisual(.9,small),full_small=full.addVisual(.9,small);
    require(lag.state().minus(full.state()).norm()<1e-8,"marginalized reference response disagrees to first order");
    require((lag_small.covariance-full_small.covariance).norm()<1e-6,"small reference response changes covariance excessively");
    lag.add(1.,raw(1.),imu,origin,prior);
    PoseObservation stale;stale.origin=origin;stale.jacobian=Matrix::Zero(1,6);stale.residual=Vector::Zero(1);
    stale.references.push_back({0,origin,Matrix::Zero(1,6)});
    bool expired=false;try {lag.addVisual(1.,stale);} catch(const std::invalid_argument&) {expired=true;}
    require(expired,"image accepted a reference outside the retained window");
    // Keep the immediately previous non-keyframe only until its image factor
    // is fused. Compare every marginal against a batch retaining every pose.
    // Empty observations and missing images must also leave a bounded graph.
    Config sparse_config=c;sparse_config.window_sec=.41;sparse_config.keyframe_sec=.3;sparse_config.max_nodes=4;
    Config batch_config=c;batch_config.window_sec=3.;batch_config.max_nodes=32;
    Window sparse(sparse_config),batch(batch_config),missing(sparse_config);
    std::vector<ImuSample> long_imu;
    for(int k=0;k<=400;++k) long_imu.push_back({k*.005,V3::Zero(),V3(0,0,9.81)});
    for(int k=1;k<=12;++k) {
      const double t=k*.1;
      sparse.add(t,raw(t),long_imu,origin,prior,{},true);
      batch.add(t,raw(t),long_imu,origin,prior);
      require(sparse.nodes().size()<=5,"pending image made the window unbounded");
      PoseObservation pair;pair.origin=origin;pair.jacobian.resize(0,6);pair.residual.resize(0);
      if(k>1) {
        const auto previous_id=static_cast<uint64_t>(k-2);
        require(std::any_of(sparse.nodes().begin(),sparse.nodes().end(),[&](const Node& n){return n.id==previous_id;}),
                "previous image pose was removed before its likelihood");
        if(k!=5) {
          pair.jacobian=Matrix::Zero(1,6);pair.jacobian(0,3)=10.;pair.residual=Vector::Zero(1);
          pair.references.push_back({previous_id,origin,-pair.jacobian});
        }
      }
      const auto a=sparse.addVisual(t,pair),b=batch.addVisual(t,pair);
      require((a.covariance-b.covariance).norm()<1e-8,"deferred image marginal lost/doubled relative information");
      require(sparse.state().minus(batch.state()).norm()<1e-10,"deferred image marginal changed zero-innovation mean");
      require(sparse.nodes().size()<=4,"image did not trim the temporary pose");
      for(size_t i=0;i+1<sparse.nodes().size();++i)
        require(sparse.nodes()[i].keyframe,"image left an old non-keyframe alive");
      bool repeated=false;
      try {sparse.addVisual(t,pair);} catch(const std::invalid_argument&) {repeated=true;}
      require(repeated,"image was counted again after its reference entered the prior");
    }
    for(int k=1;k<=20;++k) {
      const double t=k*.1;missing.add(t,raw(t),long_imu,origin,prior,{},true);
      require(missing.nodes().size()<=5,"missing images retained an unbounded history");
    }
    std::cout<<"temporal visual likelihood, covariance, persistence, deferred references and rejection tests passed\n";
  } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}

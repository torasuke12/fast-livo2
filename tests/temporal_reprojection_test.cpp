#include "temporal_reprojection.h"
#include <Eigen/QR>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <stdexcept>
using namespace temporal_lio;
static void require(bool ok,const char* message) {if(!ok) throw std::runtime_error(message);}
int main() {
  try {
    M3 k=M3::Identity();k(0,0)=250;k(1,1)=250;k(0,2)=200;k(1,2)=150;
    const M3 r=expSO3(V3(.1,-.2,.03));const V3 t(.03,-.1,.07),world(.4,.2,4.);
    State state;state.position=V3(.2,-.1,.3);state.rotation=expSO3(V3(.2,-.1,.07));
    const auto jacobian=ReprojectionTracker::projectionJacobian(state,world,k,r,t);
    auto project=[&](const State& s) {
      const V3 p=k*(r*s.rotation.transpose()*(world-s.position)+t);
      return Eigen::Vector2d(p.x()/p.z(),p.y()/p.z());
    };
    Eigen::Matrix<double,2,6> numerical;
    for(int column=0;column<6;++column) {
      V15 d=V15::Zero();d[column]=1e-6;
      numerical.col(column)=(project(state.plus(d))-project(state.plus(-d)))/(2e-6);
    }
    require((jacobian-numerical).norm()<1e-6,"Reprojection pose derivative disagrees with finite differences");

    // Two textured surfaces at different measured depths. Render a known
    // camera translation and track pixels, with no estimated motion supplied.
    cv::Mat image(300,400,CV_8UC1);cv::RNG rng(42);rng.fill(image,cv::RNG::UNIFORM,0,256);
    cv::GaussianBlur(image,image,cv::Size(3,3),.7);
    cv::Mat moved=image.clone();
    const double motion=.02;
    for(int half=0;half<2;++half) {
      const cv::Rect region(200*half,0,200,300);
      const double z=half ? 5. : 3.;
      cv::Mat warped;
      const cv::Mat affine=(cv::Mat_<double>(2,3)<<1,0,-250*motion/z,0,1,0);
      cv::warpAffine(image,warped,affine,image.size(),cv::INTER_LINEAR,cv::BORDER_REFLECT);
      warped(region).copyTo(moved(region));
    }
    Node reference;reference.id=7;reference.time=1.;
    for(int y=40;y<=260;y+=20) for(int x=40;x<=360;x+=20) {
      if(x>170 && x<230) continue;
      const double z=x<200 ? 3. : 5.;reference.points.emplace_back((x-200)*z/250.,(y-150)*z/250.,z);
    }
    reference.points.emplace_back(0,0,-3);
    reference.points.emplace_back(std::numeric_limits<double>::quiet_NaN(),0,2);
    Node current;current.id=8;current.time=1.2;current.state.position.x()=.04;
    ReprojectionTracker tracker(k,M3::Identity(),V3::Zero());tracker.seed(image,reference);
    cv::theRNG().state=1234;
    const auto observation=tracker.observe(moved,{reference,current});
    require(cv::theRNG().state==1234,"Tracker changed another OpenCV consumer's RNG");
    require(tracker.statistics().inliers>=100,"Known image translation did not produce enough inliers");
    require(observation.references.size()==1 && observation.references[0].node==reference.id,
            "Tracked points lost their retained reference pose");
    require(observation.jacobian.rows()>0,"Known image translation produced no observation");
    const Vector correction=observation.jacobian.colPivHouseholderQr().solve(-observation.residual);
    require(std::abs(correction[3]+.02)<.005 && correction.head<3>().norm()<.003 &&
            correction.tail<2>().norm()<.005,"Tracked observation has the wrong translation sign/scale");
    Matrix joint(observation.jacobian.rows(),12);joint<<observation.jacobian,observation.references[0].jacobian;
    Matrix gauge=Matrix::Zero(12,6);
    for(int i=0;i<2;++i) {
      const auto& s=i==0 ? current.state : reference.state;
      gauge.block<3,3>(6*i,0)=s.rotation.transpose();
      gauge.block<3,3>(6*i+3,0)=-skew(s.position);
      gauge.block<3,3>(6*i+3,3).setIdentity();
    }
    require((joint*gauge).norm()<1e-8,"Relative tracks invented absolute pose information");
    // Corridor travel is approximately along the camera optical axis. Test
    // this depth-sensitive expansion as well as the sideways translation.
    cv::Mat forward=image.clone();
    for(int half=0;half<2;++half) {
      const double z=half ? 5. : 3.,scale=z/(z-.05);
      const cv::Mat affine=(cv::Mat_<double>(2,3)<<scale,0,200*(1-scale),0,scale,150*(1-scale));
      cv::Mat warped;cv::warpAffine(image,warped,affine,image.size(),cv::INTER_LINEAR,cv::BORDER_REFLECT);
      const cv::Rect region(200*half,0,200,300);warped(region).copyTo(forward(region));
    }
    current.state.position=V3(0,0,.1);
    const auto forward_observation=tracker.observe(forward,{reference,current});
    require(tracker.statistics().inliers>=100 && forward_observation.jacobian.rows()>0,
            "Optical-axis motion did not produce enough valid tracks");
    const Vector forward_correction=forward_observation.jacobian.colPivHouseholderQr().solve(-forward_observation.residual);
    require(std::abs(forward_correction[5]+.05)<.007 && forward_correction.head<3>().norm()<.004,
            "Optical-axis translation has the wrong sign/scale");
    require(tracker.observe(moved,{current}).jacobian.rows()==0,"Expired reference became an exact map point");
    current.time=1.6;
    require(tracker.observe(moved,{reference,current}).jacobian.rows()==0,"Long image gap was silently fused");
    cv::Mat blank=cv::Mat::zeros(image.size(),CV_8UC1);tracker.seed(blank,reference);current.time=1.2;
    require(tracker.observe(blank,{reference,current}).jacobian.rows()==0,"Textureless frames added information");
    bool invalid=false;try {tracker.seed(cv::Mat(),reference);} catch(const std::invalid_argument&) {invalid=true;}
    require(invalid,"Empty image accepted");
    std::cout<<"Causal tracked reprojection derivatives, motion, gauge and missing-reference tests passed\n";
    return 0;
  } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}

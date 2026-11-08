#include "temporal_reprojection.h"
#include <Eigen/Cholesky>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <algorithm>
#include <numeric>
#include <set>
#include <stdexcept>

namespace temporal_lio {
namespace {
Eigen::Matrix<double,2,3> cameraDerivative(const V3& p,const M3& k) {
  Eigen::Matrix<double,2,3> j;
  j<<k(0,0)/p.z(),0,-k(0,0)*p.x()/(p.z()*p.z()),
     0,k(1,1)/p.z(),-k(1,1)*p.y()/(p.z()*p.z());
  return j;
}
void validImage(const cv::Mat& image) {
  if(image.empty() || image.type()!=CV_8UC1)
    throw std::invalid_argument("Reprojection tracker requires an 8-bit gray image");
}
}
ReprojectionTracker::ReprojectionTracker(const M3& k,const M3& r,const V3& p,double pixel,double range)
  :k_(k),rci_(r),pci_(p),pixel_sigma_(pixel),range_sigma_(range) {
  if(!k.allFinite() || !r.allFinite() || !p.allFinite() || k(0,0)<=0 || k(1,1)<=0 ||
     !std::isfinite(pixel+range) || pixel<=0 || range<0 ||
     (r.transpose()*r-M3::Identity()).norm()>1e-6 || r.determinant()<0)
    throw std::invalid_argument("Invalid reprojection camera/noise model");
}
Eigen::Matrix<double,2,6> ReprojectionTracker::projectionJacobian(const State& s,const V3& world,
    const M3& k,const M3& r,const V3& p) {
  const V3 body=s.rotation.transpose()*(world-s.position),camera=r*body+p;
  if(!camera.allFinite() || camera.z()<=1e-6) throw std::invalid_argument("Invalid projection depth");
  const auto j=cameraDerivative(camera,k);
  Eigen::Matrix<double,2,6> out;
  out.leftCols<3>()=j*r*skew(body);out.rightCols<3>()=-j*r*s.rotation.transpose();return out;
}
void ReprojectionTracker::seed(const cv::Mat& gray,const Node& node) {
  validImage(gray);
  previous_image_=gray.clone();previous_time_=node.time;previous_id_=node.id;
  pixels_.clear();body_points_.clear();
  cv::Mat scores;cv::cornerMinEigenVal(gray,scores,7);
  double maximum=0;cv::minMaxLoc(scores,nullptr,&maximum);
  struct Candidate { size_t index; cv::Point2f pixel; float score; };
  std::vector<Candidate> candidates;
  for(size_t i=0;i<node.points.size();++i) {
    const V3 p=rci_*node.points[i]+pci_;
    if(!p.allFinite() || p.z()<=.3 || p.z()>=30) continue;
    const V3 pixel=k_*p;
    const double x=pixel.x()/pixel.z(),y=pixel.y()/pixel.z();
    if(x<=20 || y<=20 || x>=gray.cols-20 || y>=gray.rows-20) continue;
    const float score=scores.at<float>(cvRound(y),cvRound(x));
    if(score>std::max(1e-5,.005*maximum)) candidates.push_back({i,cv::Point2f(x,y),score});
  }
  std::stable_sort(candidates.begin(),candidates.end(),[](const Candidate& a,const Candidate& b){return a.score>b.score;});
  std::set<std::pair<int,int>> occupied;
  for(const auto& p:candidates) {
    if(!occupied.emplace(static_cast<int>(p.pixel.x/8),static_cast<int>(p.pixel.y/8)).second) continue;
    pixels_.push_back(p.pixel);body_points_.push_back(node.points[p.index]);
    if(pixels_.size()>=500) break;
  }
}
PoseObservation ReprojectionTracker::observe(const cv::Mat& gray,const std::vector<Node>& nodes) {
  validImage(gray);if(nodes.empty()) throw std::invalid_argument("Reprojection tracker needs a current state");
  stats_={};stats_.selected=pixels_.size();
  PoseObservation empty;empty.origin=nodes.back().state;empty.jacobian.resize(0,6);empty.residual.resize(0);
  const auto reference=std::find_if(nodes.begin(),nodes.end(),[&](const Node& n){return n.id==previous_id_;});
  if(previous_image_.empty() || pixels_.size()<30 || reference==nodes.end() || previous_id_==nodes.back().id) return empty;
  stats_.dt=nodes.back().time-previous_time_;
  if(stats_.dt<=0 || stats_.dt>.35 || gray.size()!=previous_image_.size()) return empty;
  std::vector<cv::Point2f> tracked,back;std::vector<uchar> forward_ok,back_ok;std::vector<float> errors,back_errors;
  const cv::TermCriteria termination(cv::TermCriteria::COUNT|cv::TermCriteria::EPS,40,.001);
  cv::calcOpticalFlowPyrLK(previous_image_,gray,pixels_,tracked,forward_ok,errors,cv::Size(21,21),4,termination);
  cv::calcOpticalFlowPyrLK(gray,previous_image_,tracked,back,back_ok,back_errors,cv::Size(21,21),4,termination);
  std::vector<cv::Point3d> points;std::vector<cv::Point2f> measured;std::vector<size_t> indices;
  for(size_t i=0;i<pixels_.size();++i) {
    if(!forward_ok[i] || !back_ok[i] || !std::isfinite(errors[i]) || errors[i]>=20 ||
       cv::norm(back[i]-pixels_[i])>=.6 || !std::isfinite(tracked[i].x+tracked[i].y) ||
       tracked[i].x<=10 || tracked[i].y<=10 || tracked[i].x>=gray.cols-10 || tracked[i].y>=gray.rows-10) continue;
    const V3 p=rci_*body_points_[i]+pci_;points.emplace_back(p.x(),p.y(),p.z());measured.push_back(tracked[i]);indices.push_back(i);
  }
  stats_.tracked=points.size();if(points.size()<30) return empty;
  cv::Mat camera=(cv::Mat_<double>(3,3)<<k_(0,0),0,k_(0,2),0,k_(1,1),k_(1,2),0,0,1);
  cv::Mat rv,tv,inliers;
  // Fix and restore the local-thread RNG so unrelated OpenCV users do not
  // change RANSAC's decisions or inherit this diagnostic sampling sequence.
  const uint64_t saved_rng=cv::theRNG().state;cv::theRNG().state=42;
  bool success=false;
  try {success=cv::solvePnPRansac(points,measured,camera,cv::noArray(),rv,tv,false,300,1.5,.999,inliers,cv::SOLVEPNP_EPNP);}
  catch(...) {cv::theRNG().state=saved_rng;throw;}
  cv::theRNG().state=saved_rng;
  stats_.inliers=inliers.rows;
  if(!success || inliers.rows<30 || inliers.rows<.6*points.size() || !cv::checkRange(rv) ||
     !cv::checkRange(tv) || cv::norm(rv)>.7853981633974483 || cv::norm(tv)>2.) return empty;
  Matrix current=Matrix::Zero(2*inliers.rows,6),anchor=current;Vector residual=Vector::Zero(current.rows());
  const auto& s=nodes.back().state;const auto& a=reference->state;
  int row=0;
  for(int j=0;j<inliers.rows;++j) {
    const int index=inliers.at<int>(j);const V3& body=body_points_.at(indices.at(index));
    const V3 world=a.rotation*body+a.position;
    const V3 p=rci_*s.rotation.transpose()*(world-s.position)+pci_;
    if(!p.allFinite() || p.z()<=.3) continue;
    const auto derivative=cameraDerivative(p,k_);
    const V3 reference_ray=(rci_*body+pci_).normalized();
    const Eigen::Vector2d range_j=derivative*rci_*s.rotation.transpose()*a.rotation*rci_.transpose()*reference_ray;
    const Eigen::Matrix2d covariance=pixel_sigma_*pixel_sigma_*Eigen::Matrix2d::Identity()+range_sigma_*range_sigma_*range_j*range_j.transpose();
    Eigen::LLT<Eigen::Matrix2d> whitening(covariance);
    const auto jacobian=projectionJacobian(s,world,k_,rci_,pci_);
    const Eigen::Vector2d error(k_(0,0)*p.x()/p.z()+k_(0,2)-measured[index].x,
                               k_(1,1)*p.y()/p.z()+k_(1,2)-measured[index].y);
    current.middleRows(row,2)=whitening.matrixL().solve(jacobian);
    anchor.middleRows(row,2)=whitening.matrixL().solve(pointAnchorJacobian(a,world,jacobian.rightCols<3>()));
    residual.segment<2>(row)=whitening.matrixL().solve(error);row+=2;
  }
  if(row<60) return empty;
  current.conservativeResize(row,6);anchor.conservativeResize(row,6);residual.conservativeResize(row);
  return photometricObservation(s,current,residual,1.,{{reference->id,a,anchor}});
}
} // namespace temporal_lio

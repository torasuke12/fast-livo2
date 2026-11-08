#include "temporal_lio.h"
#include <Eigen/Cholesky>
#include <Eigen/QR>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace temporal_lio {
Window::Window(Config config):config_(std::move(config)) {config_.validate();}

Report Window::add(double time,const std::vector<Return>& points,const std::vector<ImuSample>& imu,
                   const State& initial,const M15& initial_covariance,
                   const std::vector<V3>& initialization_reference,bool retain_previous_for_image) {
  const auto begin=std::chrono::steady_clock::now();
  if(!std::isfinite(time) || !initial.finite() || points.empty() ||
     (!nodes_.empty() && time<=nodes_.back().time))
    throw std::runtime_error("Temporal LIO: invalid scan/state/time");
  if(!initialization_reference.empty() && !nodes_.empty())
    throw std::invalid_argument("Temporal LIO: bootstrap reference is only allowed on the first node");
  for(const auto& p:initialization_reference)
    if(!p.allFinite()) throw std::invalid_argument("Temporal LIO: invalid bootstrap reference");
  Node next; next.id=next_id_++; next.time=time; next.raw=points; next.state=initial;
  next.pose_covariance=initial_covariance.topLeftCorner<6,6>();next.covariance_rotation=initial.rotation;
  next.initialization_reference=initialization_reference;
  double scan_start=time;
  for(const auto& p:points) scan_start=std::min(scan_start,p.time);
  // Store only this scan's raw IMU, with boundaries. No discarded history is
  // needed to re-deskew any retained scan after its velocity or biases change.
  next.scan_imu=interval(imu,scan_start,time,config_.max_imu_gap);
  next.keyframe=nodes_.empty() || time-last_keyframe_time_>=config_.keyframe_sec-1e-6;
  if(next.keyframe) last_keyframe_time_=time;
  Report report;
  if(nodes_.empty()) {
    if(!initial_covariance.allFinite() || (initial_covariance-initial_covariance.transpose()).norm()>1e-6)
      throw std::runtime_error("Temporal LIO: invalid initial covariance");
    Eigen::LLT<M15> solve(initial_covariance);
    if(solve.info()!=Eigen::Success) throw std::runtime_error("Temporal LIO: nonpositive initial covariance");
    prior_jacobian_=solve.matrixL().solve(M15::Identity()); prior_residual_=V15::Zero(); prior_origins_={initial};
    nodes_.push_back(std::move(next)); refresh(); report.status="initialized"; report.covariance=initial_covariance;
    report.information=solve.solve(M15::Identity());
    if(config_.persistent_map) insertMap(nodes_.front());
    if(config_.structural_landmarks) {optimize(report);marginalCovariance(report);}
  } else {
    auto samples=interval(imu,nodes_.back().time,time,config_.max_imu_gap);
    for(size_t i=1;i<samples.size();++i) report.max_imu_gap=std::max(report.max_imu_gap,samples[i].time-samples[i-1].time);
    auto delta=preintegrate(samples,nodes_.back().state,config_);
    report.saturated_imu_intervals=delta.saturated_intervals;
    next.state=predict(nodes_.back().state,delta,config_.gravity);
    const State predicted=next.state;
    imu_edges_.push_back({nodes_.back().id,next.id,std::move(samples),delta});
    // One target per source scan avoids counting each return against multiple
    // copies of a surface. The target is a retained, jointly optimized keyframe.
    auto target=std::find_if(nodes_.rbegin(),nodes_.rend(),[](const Node& n){return n.keyframe;});
    if(target==nodes_.rend()) throw std::logic_error("Temporal LIO: missing keyframe");
    if(!config_.structural_landmarks) lidar_edges_.push_back({next.id,target->id,{}});
    nodes_.push_back(std::move(next));
    optimize(report);
    for(const auto& edge:lidar_edges_) if(edge.source==nodes_.back().id) {
      const auto& s=nodes_.back();const bool fixed=edge.target==map_target_;
      for(const auto& m:edge.matches) {
        V3 q=s.state.rotation*s.points[m.source]+s.state.position;
        if(!fixed) {const auto& t=nodes_.at(index(edge.target));q=t.state.rotation.transpose()*(q-t.state.position);}
        report.lidar_rms+=std::pow(m.plane.normal.dot(q-m.plane.center),2); ++report.matches;
        if(fixed) ++report.map_matches;
      }
    }
    for(const auto& m:nodes_.back().landmark_matches) {
      const auto& s=nodes_.back();const auto& landmark=landmarks_.at(m.landmark);
      const double r=landmark.normal.dot(s.state.rotation*s.points[m.source]+
          (s.state.position-nodes_.at(index(plane_anchor_id_)).state.position))-landmark.distance;
      report.lidar_rms+=r*r;++report.matches;
    }
    report.lidar_rms=std::sqrt(report.lidar_rms/std::max(1,report.matches));
    for(const auto& e:imu_edges_) {
      V15 r=imuResidual(nodes_.at(index(e.a)).state,nodes_.at(index(e.b)).state,e.delta,config_.gravity);
      report.imu_rms+=r.dot(e.delta.information*r);
    }
    report.imu_edges=imu_edges_.size(); report.imu_rms=std::sqrt(report.imu_rms/std::max(1,15*report.imu_edges));
    report.correction=nodes_.back().state.minus(predicted);
    report.status=report.matches ? "updated" : "imu_only";
    // The consecutive image factor needs the previous pose until addVisual.
    // Keep at most one extra pose temporarily; older non-keyframes still leave
    // even if images are missing. This does not promote every image to a keyframe.
    report.marginalized+=trim(retain_previous_for_image);
    marginalCovariance(report);
  }
  nodes_.back().pose_covariance=report.covariance.topLeftCorner<6,6>();
  nodes_.back().covariance_rotation=nodes_.back().state.rotation;
  report.map_planes=map_planes_.size();report.landmarks=landmarks_.size();
  for(const auto& n:nodes_) if(!n.state.finite()) throw std::runtime_error("Temporal LIO: invalid optimized state");
  report.nodes=nodes_.size(); report.span=nodes_.back().time-nodes_.front().time;
  report.solve_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
  return report;
}

int Window::trim(bool retain_previous_for_image) {
  int removed=0;
  const uint64_t previous=nodes_.size()>1 ? nodes_[nodes_.size()-2].id : 0;
  for(int i=static_cast<int>(nodes_.size())-2;i>=0;--i) {
    if(retain_previous_for_image && nodes_[i].id==previous) continue;
    if(!nodes_[i].keyframe) {marginalize(i);++removed;}
  }
  const size_t limit=config_.max_nodes+(retain_previous_for_image ? 1 : 0);
  while(nodes_.size()>1 && (nodes_.back().time-nodes_.front().time>config_.window_sec+1e-6 ||
                           nodes_.size()>limit)) {
    if(retain_previous_for_image && nodes_.size()==2) break;
    marginalize(0);++removed;
  }
  return removed;
}

void Window::optimize(Report& report) {
  for(int iteration=0;iteration<config_.iterations;++iteration) {
    refresh();
    Matrix jacobian;Vector residual;double cost;linearize(jacobian,residual,cost);
    report.cost_before=cost;
    const Vector scale=jacobian.colwise().norm().transpose().cwiseMax(1.0).cwiseInverse();
    const Matrix scaled=jacobian*scale.asDiagonal();
    std::vector<State> saved; for(const auto& n:nodes_) saved.push_back(n.state);
    const auto saved_landmarks=landmarks_;
    bool accepted=false;
    for(double damping:{0.0,1e-6,1e-4,1e-2}) {
      const int extra=damping>0 ? scaled.cols() : 0;
      Matrix damped=Matrix::Zero(scaled.rows()+extra,scaled.cols());
      damped.topRows(scaled.rows())=scaled;
      if(extra) damped.bottomRows(extra).diagonal().array()=std::sqrt(damping);
      Vector rhs=Vector::Zero(damped.rows());rhs.head(residual.size())=-residual;
      Vector solution;bool solved=false;
      if(config_.structural_landmarks && damped.rows()>=damped.cols()) {
        // Blocked, non-pivoting QR is much faster for the larger plane/state
        // system. Keep square-root arithmetic; use pivoting near a small
        // diagonal rather than forming normal equations or adding information.
        Eigen::HouseholderQR<Matrix> qr(damped);
        if((qr.matrixQR().diagonal().array().abs()>1e-10).all()) {
          solution=qr.solve(rhs);solved=true;
        }
      }
      if(!solved) {
        Eigen::ColPivHouseholderQR<Matrix> solve(damped);
        if(solve.rank()!=scaled.cols()) continue;
        solution=solve.solve(rhs);
      }
      const Vector dx=scale.asDiagonal()*solution;
      if(!dx.allFinite()) continue;
      for(double step:{1.0,0.5,0.25,0.125}) {
        for(size_t i=0;i<nodes_.size();++i) nodes_[i].state=saved[i].plus(step*dx.segment<15>(15*i));
        for(size_t i=0;i<landmarks_.size();++i) {
          const auto& saved=saved_landmarks[i];const int col=15*nodes_.size()+3*i;
          landmarks_[i].normal=(saved.normal+planeBasis(saved.normal)*(step*dx.segment<2>(col))).normalized();
          landmarks_[i].distance=saved.distance+step*dx[col+2];
        }
        double trial_cost;
        if(config_.consistent_deskew || config_.structure_enabled) trial_cost=trialCost();
        else {Matrix trial_j;Vector trial_r;linearize(trial_j,trial_r,trial_cost);}
        if(trial_cost<=cost+1e-8*std::max(1.0,std::abs(cost))) {
          report.cost_after=trial_cost; accepted=true; break;
        }
      }
      if(accepted) break;
    }
    ++report.iterations;
    if(!accepted) {
      for(size_t i=0;i<nodes_.size();++i) nodes_[i].state=saved[i];
      landmarks_=saved_landmarks;
      report.cost_after=cost; break;
    }
  }
  refresh();
}

void Window::marginalCovariance(Report& report) const {
  Matrix jacobian;Vector residual;double cost;linearize(jacobian,residual,cost);
  const int dim=jacobian.cols();
  const Vector scale=jacobian.colwise().norm().transpose().cwiseMax(1.0).cwiseInverse();
  const int latest=15*(nodes_.size()-1);
  Matrix select=Matrix::Zero(dim,15);select.middleRows(latest,15)=scale.segment(latest,15).asDiagonal();
  Matrix covariance_root;bool solved=false;
  const Matrix scaled=jacobian*scale.asDiagonal();
  if(config_.structural_landmarks && scaled.rows()>=dim) {
    Eigen::HouseholderQR<Matrix> qr(scaled);
    if((qr.matrixQR().diagonal().array().abs()>1e-10).all()) {
      const Matrix triangular=qr.matrixQR().topLeftCorner(dim,dim).triangularView<Eigen::Upper>();
      covariance_root=triangular.transpose().triangularView<Eigen::Lower>().solve(select);solved=true;
    }
  }
  if(!solved) {
    Eigen::ColPivHouseholderQR<Matrix> qr(scaled);
    if(qr.rank()!=dim) throw std::runtime_error("Temporal LIO: joint square-root system lost rank");
    const Matrix triangular=qr.matrixR().topLeftCorner(dim,dim).triangularView<Eigen::Upper>();
    const Matrix projected=qr.colsPermutation().transpose()*select;
    covariance_root=triangular.transpose().triangularView<Eigen::Lower>().solve(projected);
  }
  report.covariance=covariance_root.transpose()*covariance_root;
  report.covariance=0.5*(report.covariance+report.covariance.transpose()).eval();
  Eigen::LLT<M15> marginal(report.covariance);
  if(!report.covariance.allFinite() || marginal.info()!=Eigen::Success)
    throw std::runtime_error("Temporal LIO: invalid marginal covariance");
  report.information=marginal.solve(M15::Identity());
}
} // namespace temporal_lio

#include "temporal_lio.h"
#include <Eigen/Cholesky>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace temporal_lio {
M3 skew(const V3& x) {
  M3 m; m << 0, -x.z(), x.y(), x.z(), 0, -x.x(), -x.y(), x.x(), 0; return m;
}
M3 expSO3(const V3& x) {
  const double n = x.norm();
  if (n < 1e-8) return M3::Identity() + skew(x) + 0.5 * skew(x) * skew(x);
  return Eigen::AngleAxisd(n, x / n).toRotationMatrix();
}
V3 logSO3(const M3& r) { Eigen::AngleAxisd a(r); return a.angle() * a.axis(); }
M3 rightJacobianInverse(const V3& x) {
  const double n = x.norm();
  const double c = n < 1e-5 ? 1.0/12.0 : (1.0 - 0.5*n/std::tan(0.5*n))/(n*n);
  return M3::Identity() + 0.5 * skew(x) + c * skew(x) * skew(x);
}
State State::plus(const V15& d) const {
  State s = *this; s.rotation *= expSO3(d.head<3>()); s.position += d.segment<3>(3);
  s.velocity += d.segment<3>(6); s.gyro_bias += d.segment<3>(9); s.accel_bias += d.tail<3>(); return s;
}
V15 State::minus(const State& o) const {
  V15 d; d << logSO3(o.rotation.transpose()*rotation), position-o.position,
    velocity-o.velocity, gyro_bias-o.gyro_bias, accel_bias-o.accel_bias; return d;
}
bool State::finite() const {
  return rotation.allFinite() && position.allFinite() && velocity.allFinite() &&
    gyro_bias.allFinite() && accel_bias.allFinite() &&
    (rotation.transpose()*rotation-M3::Identity()).norm()<1e-6 && rotation.determinant()>0;
}
void Config::validate() const {
  if(!std::isfinite(visual_degeneracy_ratio) || visual_degeneracy_ratio<=0 || visual_degeneracy_ratio>=1)
    throw std::invalid_argument("Invalid temporal visual degeneracy ratio");
  if(!std::isfinite(accel_saturation) || accel_saturation<0 ||
     !std::isfinite(accel_saturation_variance) || accel_saturation_variance<=0)
    throw std::invalid_argument("Invalid temporal IMU saturation configuration");
  if(structural_landmarks && (!structure_enabled || !consistent_deskew || persistent_map || max_landmarks<3 || max_landmarks>512))
    throw std::invalid_argument("Structural landmarks require structure enabled, persistent_map disabled and 3..512 landmarks");
  if(map_max_planes<30) throw std::invalid_argument("Temporal map needs at least 30 planes");
  if(structure_enabled && (!(structure_sigma>0 && structure_max_angle>0 && structure_max_angle<.3 && structure_min_planes>=3)
      || !std::isfinite(structure_sigma+structure_max_angle) || gravity.norm()<1e-6))
    throw std::invalid_argument("Invalid temporal structural orientation configuration");
  if (!(window_sec > 0 && keyframe_sec > 0 && keyframe_sec <= window_sec &&
        max_nodes >= 3 && max_nodes <= 64 && iterations >= 1 && iterations <= 20 &&
        max_source_points >= 30 && voxel_size > 0 && sample_size > 0 &&
        min_plane_points >= 3 && plane_ratio > 0 && plane_ratio < 1 && min_plane_spread > 0 &&
        max_residual > 0 && point_sigma > 0 && huber_delta > 0 && max_imu_gap > 0) ||
      !std::isfinite(window_sec + keyframe_sec + voxel_size + sample_size + plane_ratio +
                     min_plane_spread + max_residual + point_sigma + huber_delta + max_imu_gap) ||
      !gravity.allFinite() || !lidar_to_imu_translation.allFinite() ||
      !lidar_to_imu_rotation.allFinite() ||
      (lidar_to_imu_rotation.transpose()*lidar_to_imu_rotation-M3::Identity()).norm()>1e-6 ||
      lidar_to_imu_rotation.determinant()<0 ||
      !gyro_variance.allFinite() || !accel_variance.allFinite() ||
      !gyro_bias_variance.allFinite() || !accel_bias_variance.allFinite() ||
      (gyro_variance.array()<=0).any() || (accel_variance.array()<=0).any() ||
      (gyro_bias_variance.array()<=0).any() || (accel_bias_variance.array()<=0).any())
    throw std::invalid_argument("Invalid temporal LIO configuration");
}

std::vector<ImuSample> interval(const std::vector<ImuSample>& s, double begin, double end, double max_gap) {
  if (!std::isfinite(begin) || !std::isfinite(end) || !(end > begin) || s.empty())
    throw std::runtime_error("Temporal LIO: invalid/empty IMU interval");
  for (size_t i=0; i<s.size(); ++i) {
    if (!std::isfinite(s[i].time) || !s[i].gyro.allFinite() || !s[i].accel.allFinite() ||
        (i && s[i].time <= s[i-1].time)) throw std::runtime_error("Temporal LIO: nonfinite or unordered IMU");
  }
  auto sample = [&](double t) {
    auto it = std::lower_bound(s.begin(), s.end(), t, [](const ImuSample& a, double b){return a.time < b;});
    ImuSample q;
    if (it == s.begin()) {
      if (it->time-t > max_gap) throw std::runtime_error("Temporal LIO: missing IMU start boundary");
      q=*it;
    } else if (it == s.end()) {
      if (t-s.back().time > max_gap) throw std::runtime_error("Temporal LIO: missing IMU end boundary");
      q=s.back();
    } else {
      const auto& a=*(it-1); const auto& b=*it;
      if (b.time-a.time > max_gap) throw std::runtime_error("Temporal LIO: IMU gap at boundary");
      double u=(t-a.time)/(b.time-a.time); q.gyro=(1-u)*a.gyro+u*b.gyro; q.accel=(1-u)*a.accel+u*b.accel;
      q.accel_max_abs=a.accel.cwiseAbs().cwiseMax(b.accel.cwiseAbs())
          .cwiseMax(a.accel_max_abs).cwiseMax(b.accel_max_abs);
    }
    q.time=t; return q;
  };
  std::vector<ImuSample> result{sample(begin)};
  for (const auto& q:s) if(q.time>begin && q.time<end) result.push_back(q);
  result.push_back(sample(end));
  for(size_t i=1;i<result.size();++i) if(result[i].time-result[i-1].time>max_gap)
    throw std::runtime_error("Temporal LIO: IMU gap exceeds configured limit");
  return result;
}

static Preintegration integrate(const std::vector<ImuSample>& samples, const State& state,
                                const Config& c, bool covariance) {
  if (samples.size()<2) throw std::runtime_error("Temporal LIO: preintegration needs two boundaries");
  Preintegration d; d.gyro_bias=state.gyro_bias; d.accel_bias=state.accel_bias;
  for(size_t i=1;i<samples.size();++i) {
    const double dt=samples[i].time-samples[i-1].time;
    if (!(dt>0 && dt<=c.max_imu_gap)) throw std::runtime_error("Temporal LIO: invalid integration step");
    const V3 w=0.5*(samples[i-1].gyro+samples[i].gyro)-state.gyro_bias;
    const V3 a=0.5*(samples[i-1].accel+samples[i].accel)-state.accel_bias;
    const M3 rmid=d.rotation*expSO3(w*(0.5*dt));
    if(covariance) {
      M15 f=M15::Identity();
      f.block<3,3>(0,0)=expSO3(-w*dt); f.block<3,3>(0,9)=-M3::Identity()*dt;
      f.block<3,3>(3,0)=-0.5*rmid*skew(a)*dt*dt;
      f.block<3,3>(3,6)=M3::Identity()*dt; f.block<3,3>(3,12)=-0.5*rmid*dt*dt;
      f.block<3,3>(6,0)=-rmid*skew(a)*dt; f.block<3,3>(6,12)=-rmid*dt;
      Eigen::Matrix<double,15,12> g=Eigen::Matrix<double,15,12>::Zero();
      g.block<3,3>(0,0)=-M3::Identity()*dt;
      g.block<3,3>(3,3)=0.5*rmid*dt*dt; g.block<3,3>(6,3)=rmid*dt;
      g.block<3,3>(9,6)=M3::Identity()*dt; g.block<3,3>(12,9)=M3::Identity()*dt;
      Eigen::Matrix<double,12,1> q; q<<c.gyro_variance,c.accel_variance,c.gyro_bias_variance,c.accel_bias_variance;
      if(c.accel_saturation>0) {
        const V3 peak=samples[i-1].accel.cwiseAbs().cwiseMax(samples[i].accel.cwiseAbs())
            .cwiseMax(samples[i-1].accel_max_abs).cwiseMax(samples[i].accel_max_abs);
        bool saturated=false;
        for(int axis=0;axis<3;++axis) if(peak[axis]>=c.accel_saturation) {
          q[3+axis]=std::max(q[3+axis],c.accel_saturation_variance);saturated=true;
        }
        if(saturated) ++d.saturated_intervals;
      }
      d.covariance=(f*d.covariance*f.transpose()+g*q.asDiagonal()*g.transpose()).eval();
    }
    const V3 dv=rmid*a*dt;
    d.position+=d.velocity*dt+0.5*dv*dt; d.velocity+=dv; d.rotation*=expSO3(w*dt); d.dt+=dt;
  }
  return d;
}
Preintegration preintegrate(const std::vector<ImuSample>& samples, const State& state, const Config& c) {
  Preintegration d=integrate(samples,state,c,true);
  // Residual position and velocity are in the start frame; rotation is in the end tangent.
  const double eps=1e-5;
  for(int k=0;k<6;++k) {
    V15 dx=V15::Zero(); dx[9+k]=eps;
    const auto p=integrate(samples,state.plus(dx),c,false), m=integrate(samples,state.plus(-dx),c,false);
    d.bias_jacobian.block<3,1>(0,k)=logSO3(m.rotation.transpose()*p.rotation)/(2*eps);
    d.bias_jacobian.block<3,1>(3,k)=(p.position-m.position)/(2*eps);
    d.bias_jacobian.block<3,1>(6,k)=(p.velocity-m.velocity)/(2*eps);
  }
  d.covariance=0.5*(d.covariance+d.covariance.transpose()).eval();
  d.covariance.diagonal().array()+=1e-12; // numerical floor, not a motion constraint
  Eigen::LLT<M15> solver(d.covariance);
  if(solver.info()!=Eigen::Success) throw std::runtime_error("Temporal LIO: invalid IMU covariance");
  d.information=solver.solve(M15::Identity());
  return d;
}
State predict(const State& a, const Preintegration& d, const V3& g) {
  State b=a; b.rotation=a.rotation*d.rotation;
  b.position=a.position+a.velocity*d.dt+0.5*g*d.dt*d.dt+a.rotation*d.position;
  b.velocity=a.velocity+g*d.dt+a.rotation*d.velocity; return b;
}
V15 imuResidual(const State& a,const State& b,const Preintegration& d,const V3& g) {
  Eigen::Matrix<double,6,1> db; db<<a.gyro_bias-d.gyro_bias,a.accel_bias-d.accel_bias;
  const Eigen::Matrix<double,9,1> correction=d.bias_jacobian*db;
  V15 r;
  r<<logSO3((d.rotation*expSO3(correction.head<3>())).transpose()*a.rotation.transpose()*b.rotation),
    a.rotation.transpose()*(b.position-a.position-a.velocity*d.dt-0.5*g*d.dt*d.dt)-d.position-correction.segment<3>(3),
    a.rotation.transpose()*(b.velocity-a.velocity-g*d.dt)-d.velocity-correction.tail<3>(),
    b.gyro_bias-a.gyro_bias,b.accel_bias-a.accel_bias;
  return r;
}

Eigen::Matrix<double,15,30> imuJacobians(const State& a,const State& b,
                                       const Preintegration& d,const V3& g) {
  // World position/velocity enter affinely. Finite differences of large
  // world coordinates destroy the exact common-translation null direction.
  Eigen::Matrix<double,15,30> j=Eigen::Matrix<double,15,30>::Zero();
  Eigen::Matrix<double,6,1> db;db<<a.gyro_bias-d.gyro_bias,a.accel_bias-d.accel_bias;
  const V3 correction=d.bias_jacobian.topRows<3>()*db;
  const M3 corrected_rotation=d.rotation*expSO3(correction);
  const V3 rotation_residual=logSO3(corrected_rotation.transpose()*a.rotation.transpose()*b.rotation);
  const M3 left_inverse=rightJacobianInverse(-rotation_residual);
  const double angle=correction.norm();const M3 cross=skew(correction);
  const M3 correction_jacobian=M3::Identity()
      -(angle<1e-5 ? .5-angle*angle/24 : (1-std::cos(angle))/(angle*angle))*cross
      +(angle<1e-5 ? 1./6.-angle*angle/120 : (angle-std::sin(angle))/(angle*angle*angle))*cross*cross;
  j.block<3,3>(0,0)=-left_inverse*corrected_rotation.transpose();
  j.block<3,3>(0,15)=rightJacobianInverse(rotation_residual);
  j.block<3,6>(0,9)=-left_inverse*correction_jacobian*d.bias_jacobian.topRows<3>();
  const M3 inverse_rotation=a.rotation.transpose();
  j.block<3,3>(3,0)=skew(inverse_rotation*(b.position-a.position-a.velocity*d.dt-.5*g*d.dt*d.dt));
  j.block<3,3>(3,3)=-inverse_rotation;
  j.block<3,3>(3,18)=inverse_rotation;
  j.block<3,3>(3,6)=-d.dt*inverse_rotation;
  j.block<3,6>(3,9)=-d.bias_jacobian.middleRows<3>(3);
  j.block<3,3>(6,0)=skew(inverse_rotation*(b.velocity-a.velocity-g*d.dt));
  j.block<3,3>(6,6)=-inverse_rotation;
  j.block<3,3>(6,21)=inverse_rotation;
  j.block<3,6>(6,9)=-d.bias_jacobian.bottomRows<3>();
  j.block<6,6>(9,9)=-Eigen::Matrix<double,6,6>::Identity();
  j.block<6,6>(9,24)=Eigen::Matrix<double,6,6>::Identity();
  return j;
}

std::vector<V3> deskew(const std::vector<Return>& points, const std::vector<ImuSample>& samples,
                       double end, const State& state, const Config& c) {
  std::vector<V3> result; result.reserve(points.size());
  if(points.empty()) return result;
  double begin=end;
  for(const auto& p:points) {
    if(!p.point.allFinite() || !std::isfinite(p.time) || p.time>end+1e-6)
      throw std::runtime_error("Temporal LIO: invalid point or point timestamp");
    begin=std::min(begin,p.time);
  }
  if(begin==end) {
    for(const auto& p:points) result.push_back(c.lidar_to_imu_rotation*p.point+c.lidar_to_imu_translation);
    return result;
  }
  auto imu=interval(samples,begin,end,c.max_imu_gap);
  // Deskew produces points in the end frame. Integrate in that frame, as the
  // deskew Jacobians already do. Adding and subtracting the world position
  // introduced origin-dependent roundoff before voxelization and fitting.
  std::vector<State> trajectory(imu.size());
  trajectory.back().velocity=state.rotation.transpose()*state.velocity;
  const V3 gravity=state.rotation.transpose()*c.gravity;
  for(size_t k=imu.size()-1;k>0;--k) {
    const double dt=imu[k].time-imu[k-1].time;
    const V3 w=0.5*(imu[k].gyro+imu[k-1].gyro)-state.gyro_bias;
    const V3 a=0.5*(imu[k].accel+imu[k-1].accel)-state.accel_bias;
    const auto& next=trajectory[k]; auto& prev=trajectory[k-1]; prev=next;
    prev.rotation=next.rotation*expSO3(-w*dt);
    const V3 aw=next.rotation*expSO3(-w*dt*0.5)*a+gravity;
    prev.velocity=next.velocity-aw*dt;
    prev.position=next.position-next.velocity*dt+0.5*aw*dt*dt;
  }
  for(const auto& p:points) {
    auto it=std::upper_bound(imu.begin(),imu.end(),p.time,[](double t,const ImuSample& q){return t<q.time;});
    const size_t k=std::min(imu.size()-2, static_cast<size_t>(std::max<std::ptrdiff_t>(0,it-imu.begin()-1)));
    const double dt=imu[k+1].time-imu[k].time, t=std::clamp(p.time-imu[k].time,0.0,dt);
    const auto& a=trajectory[k]; const auto& b=trajectory[k+1];
    const M3 r=a.rotation*expSO3(logSO3(a.rotation.transpose()*b.rotation)*(t/dt));
    const V3 pos=a.position+a.velocity*t+0.5*(b.velocity-a.velocity)*(t*t/dt);
    result.push_back(r*(c.lidar_to_imu_rotation*p.point+c.lidar_to_imu_translation)+pos);
  }
  return result;
}
std::vector<Eigen::Matrix<double,3,15>> deskewJacobians(const std::vector<Return>& points,
    const std::vector<ImuSample>& samples,double end,const State& state,const Config& c) {
  using J=Eigen::Matrix<double,3,15>;
  std::vector<J> result(points.size(),J::Zero());
  double begin=end;
  for(const auto& p:points) begin=std::min(begin,p.time);
  if(begin==end) return result;
  const auto imu=interval(samples,begin,end,c.max_imu_gap);
  // Differentiate only the short IMU trajectory. Each return then needs one
  // rotation interpolation, rather than 24 complete perturbed deskews.
  // Work in the end frame to avoid subtracting large world coordinates.
  auto trajectory=[&](const State& s) {
    std::vector<State> knots(imu.size());
    knots.back().velocity=s.rotation.transpose()*s.velocity;
    const V3 gravity=s.rotation.transpose()*c.gravity;
    for(size_t k=imu.size()-1;k>0;--k) {
      const double dt=imu[k].time-imu[k-1].time;
      const V3 w=.5*(imu[k].gyro+imu[k-1].gyro)-s.gyro_bias;
      const V3 a=.5*(imu[k].accel+imu[k-1].accel)-s.accel_bias;
      const auto& next=knots[k];auto& prev=knots[k-1];
      prev.rotation=next.rotation*expSO3(-w*dt);
      const V3 aw=next.rotation*expSO3(-w*dt*.5)*a+gravity;
      prev.velocity=next.velocity-aw*dt;
      prev.position=next.position-next.velocity*dt+.5*aw*dt*dt;
    }
    return knots;
  };
  const auto base=trajectory(state);
  std::vector<J> jr(imu.size(),J::Zero()),jp=jr,jv=jr;
  constexpr double epsilon=1e-5;
  for(int k:{0,1,2,6,7,8,9,10,11,12,13,14}) {
    V15 dx=V15::Zero();dx[k]=epsilon;
    const auto plus=trajectory(state.plus(dx)),minus=trajectory(state.plus(-dx));
    for(size_t i=0;i<imu.size();++i) {
      jr[i].col(k)=logSO3(minus[i].rotation.transpose()*plus[i].rotation)/(2*epsilon);
      jp[i].col(k)=(plus[i].position-minus[i].position)/(2*epsilon);
      jv[i].col(k)=(plus[i].velocity-minus[i].velocity)/(2*epsilon);
    }
  }
  for(size_t i=0;i<points.size();++i) {
    const auto& p=points[i];
    auto it=std::upper_bound(imu.begin(),imu.end(),p.time,[](double t,const ImuSample& q){return t<q.time;});
    const size_t k=std::min(imu.size()-2,static_cast<size_t>(std::max<std::ptrdiff_t>(0,it-imu.begin()-1)));
    const double dt=imu[k+1].time-imu[k].time,t=std::clamp(p.time-imu[k].time,0.0,dt);
    const V3 w=.5*(imu[k].gyro+imu[k+1].gyro)-state.gyro_bias;
    const V3 angle=w*t,body=c.lidar_to_imu_rotation*p.point+c.lidar_to_imu_translation;
    const M3 e=expSO3(angle);
    result[i]=-base[k].rotation*skew(e*body)*jr[k]+jp[k]+jv[k]*t+.5*(jv[k+1]-jv[k])*(t*t/dt);
    const double theta=angle.norm();const M3 cross=skew(angle);
    const M3 right=M3::Identity()-(theta<1e-5 ? .5-theta*theta/24 : (1-std::cos(theta))/(theta*theta))*cross
      +(theta<1e-5 ? 1./6.-theta*theta/120 : (theta-std::sin(theta))/(theta*theta*theta))*cross*cross;
    result[i].block<3,3>(0,9)+=base[k].rotation*e*skew(body)*right*t;
  }
  return result;
}
} // namespace temporal_lio

#pragma once
#include "temporal_lio.h"
#include <opencv2/core.hpp>

namespace temporal_lio {
// A causal, consecutive-image likelihood. Points carry measured range and a
// retained acquisition pose; no PnP pose is inserted as an absolute constraint.
class ReprojectionTracker {
public:
  struct Statistics { int selected=0, tracked=0, inliers=0; double dt=0; };
  ReprojectionTracker(const M3& intrinsics, const M3& imu_to_camera_rotation,
                      const V3& imu_to_camera_translation, double pixel_sigma=1., double range_sigma=.03);
  PoseObservation observe(const cv::Mat& gray, const std::vector<Node>& nodes);
  void seed(const cv::Mat& gray, const Node& node);
  const Statistics& statistics() const { return stats_; }
  // Exposed for finite-difference and gauge regression tests.
  static Eigen::Matrix<double,2,6> projectionJacobian(const State& current,
      const V3& world, const M3& intrinsics, const M3& rotation, const V3& translation);
private:
  M3 k_,rci_; V3 pci_;
  double pixel_sigma_,range_sigma_,previous_time_=0;
  uint64_t previous_id_=UINT64_MAX;
  cv::Mat previous_image_;
  std::vector<cv::Point2f> pixels_;
  std::vector<V3> body_points_;
  Statistics stats_;
};
} // namespace temporal_lio

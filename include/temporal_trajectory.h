#pragma once

#include "temporal_lio.h"

namespace temporal_lio {

// Retained keyframes may be revised for the full lag. Non-keyframes disappear
// after the next solve, so they cannot be kept as unrevised historical poses
// between revised keyframes. Keep only a replaceable live endpoint for them.
class TrajectoryHistory {
 public:
  struct Pose {
    uint64_t id;
    double time;
    M3 rotation;
    V3 position;
  };

  void update(const std::vector<Node>& nodes) {
    if (nodes.empty()) return;
    if (has_live_tail_) {
      poses_.pop_back();
      has_live_tail_ = false;
    }
    for (const auto& node : nodes) {
      const Pose pose{node.id, node.time, node.state.rotation, node.state.position};
      if (node.keyframe) {
        const auto found = indices_.find(node.id);
        if (found == indices_.end()) {
          indices_[node.id] = poses_.size();
          poses_.push_back(pose);
        } else {
          poses_[found->second] = pose;
        }
      } else if (node.id == nodes.back().id) {
        poses_.push_back(pose);
        has_live_tail_ = true;
      }
    }
    // Older poses remain at their last reported estimate. Only poses still in
    // the estimator window can receive further corrections.
    indices_.erase(indices_.begin(), indices_.lower_bound(nodes.front().id));
  }

  const std::vector<Pose>& poses() const { return poses_; }

 private:
  std::vector<Pose> poses_;
  std::map<uint64_t, size_t> indices_;
  bool has_live_tail_ = false;
};
}  // namespace temporal_lio

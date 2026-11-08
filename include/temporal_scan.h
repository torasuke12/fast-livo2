#pragma once
#include <cmath>
#include <stdexcept>
#include <utility>

namespace temporal_lio {
// Pending curvature is relative to the preceding image time. A 10 Hz LiDAR
// packet can span several 30 Hz images: partition its remainder at every cut,
// preserving future returns rather than consuming them at the next image.
template<class Cloud>
void splitPendingScan(Cloud& pending,Cloud& current,double interval_ms) {
  if(!std::isfinite(interval_ms) || interval_ms<=0 || &pending==&current)
    throw std::invalid_argument("Temporal LIVO: invalid pending scan interval");
  auto returns=std::move(pending.points);
  pending.clear();current.clear();
  for(auto point:returns) {
    if(point.curvature<interval_ms) current.push_back(point);
    else {point.curvature-=interval_ms;pending.push_back(point);}
  }
}
} // namespace temporal_lio

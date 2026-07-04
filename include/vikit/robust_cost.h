#ifndef FAST_LIVO2_VIKIT_ROBUST_COST_H_
#define FAST_LIVO2_VIKIT_ROBUST_COST_H_

namespace vk
{
struct HuberWeightFunction
{
  explicit HuberWeightFunction(double k = 1.345) : k_(k) {}
  double weight(double e) const { return e < k_ ? 1.0 : k_ / e; }
  double k_;
};
} // namespace vk

#endif // FAST_LIVO2_VIKIT_ROBUST_COST_H_


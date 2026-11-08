#pragma once

#include <cmath>

namespace fast_livo2 {
// Bounds of every bilinear pixel read, including the central-difference halo.
// Match the float rounding used by the frontend's floorf(pixel / scale).
inline bool visualPatchInBounds(int width, int height, double u, double v,
                                int patch_size, int pyramid_level, bool gradient) {
  if (width <= 0 || height <= 0 || patch_size <= 0 || pyramid_level < 0 || pyramid_level > 29 ||
      !std::isfinite(u) || !std::isfinite(v) || u < 0 || v < 0 || u >= width || v >= height)
    return false;
  const double scale = double(1 << pyramid_level);
  const double x = std::floor(static_cast<float>(u / scale)) * scale;
  const double y = std::floor(static_cast<float>(v / scale)) * scale;
  const double left = (patch_size / 2 + (gradient ? 1 : 0)) * scale;
  const double right = (patch_size - patch_size / 2 + (gradient ? 1 : 0)) * scale;
  return x >= left && y >= left && x + right < width && y + right < height;
}
}  // namespace fast_livo2

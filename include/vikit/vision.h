#ifndef FAST_LIVO2_VIKIT_VISION_H_
#define FAST_LIVO2_VIKIT_VISION_H_

#include <opencv2/imgproc.hpp>

namespace vk
{
inline void halfSample(const cv::Mat &in, cv::Mat &out)
{
  cv::resize(in, out, cv::Size(in.cols / 2, in.rows / 2), 0.0, 0.0, cv::INTER_AREA);
}

inline float interpolateMat_8u(const cv::Mat &mat, float u, float v)
{
  const int x = static_cast<int>(std::floor(u));
  const int y = static_cast<int>(std::floor(v));
  if (x < 0 || y < 0 || x + 1 >= mat.cols || y + 1 >= mat.rows) return 0.0f;
  const float subpix_x = u - x;
  const float subpix_y = v - y;
  const float w00 = (1.0f - subpix_x) * (1.0f - subpix_y);
  const float w01 = subpix_x * (1.0f - subpix_y);
  const float w10 = (1.0f - subpix_x) * subpix_y;
  const float w11 = subpix_x * subpix_y;
  const uint8_t *ptr = mat.ptr<uint8_t>(y) + x;
  return w00 * ptr[0] + w01 * ptr[1] + w10 * ptr[mat.step] + w11 * ptr[mat.step + 1];
}

inline float shiTomasiScore(const cv::Mat &img, int u, int v)
{
  constexpr int halfbox = 4;
  if (u < halfbox || v < halfbox || u >= img.cols - halfbox || v >= img.rows - halfbox) return 0.0f;
  double dXX = 0.0;
  double dYY = 0.0;
  double dXY = 0.0;
  for (int y = v - halfbox; y <= v + halfbox; ++y)
  {
    const uint8_t *ptr = img.ptr<uint8_t>(y);
    for (int x = u - halfbox; x <= u + halfbox; ++x)
    {
      const int dx = ptr[x + 1] - ptr[x - 1];
      const int dy = img.ptr<uint8_t>(y + 1)[x] - img.ptr<uint8_t>(y - 1)[x];
      dXX += dx * dx;
      dYY += dy * dy;
      dXY += dx * dy;
    }
  }
  const double trace = dXX + dYY;
  const double det = dXX * dYY - dXY * dXY;
  return static_cast<float>(0.5 * (trace - std::sqrt(std::max(0.0, trace * trace - 4.0 * det))));
}
} // namespace vk

#endif // FAST_LIVO2_VIKIT_VISION_H_


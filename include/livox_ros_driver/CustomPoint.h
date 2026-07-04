#ifndef FAST_LIVO2_LIVOX_CUSTOM_POINT_COMPAT_H_
#define FAST_LIVO2_LIVOX_CUSTOM_POINT_COMPAT_H_

#include <cstdint>
#include <memory>

namespace livox_ros_driver
{
struct CustomPoint
{
  using SharedPtr = std::shared_ptr<CustomPoint>;
  using ConstSharedPtr = std::shared_ptr<const CustomPoint>;

  uint32_t offset_time = 0;
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
  uint8_t reflectivity = 0;
  uint8_t tag = 0;
  uint8_t line = 0;
};
} // namespace livox_ros_driver

#endif // FAST_LIVO2_LIVOX_CUSTOM_POINT_COMPAT_H_


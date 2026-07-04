#ifndef FAST_LIVO2_LIVOX_CUSTOM_MSG_COMPAT_H_
#define FAST_LIVO2_LIVOX_CUSTOM_MSG_COMPAT_H_

#include <livox_ros_driver/CustomPoint.h>
#include <std_msgs/msg/header.hpp>

#include <memory>
#include <vector>

namespace livox_ros_driver
{
struct CustomMsg
{
  using SharedPtr = std::shared_ptr<CustomMsg>;
  using ConstSharedPtr = std::shared_ptr<const CustomMsg>;

  std_msgs::msg::Header header;
  uint64_t timebase = 0;
  uint32_t point_num = 0;
  uint8_t lidar_id = 0;
  uint8_t rsvd[3] = {0, 0, 0};
  std::vector<CustomPoint> points;
};
} // namespace livox_ros_driver

#endif // FAST_LIVO2_LIVOX_CUSTOM_MSG_COMPAT_H_


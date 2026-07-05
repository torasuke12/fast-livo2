#ifndef FAST_LIVO2_ROS2_COMPAT_HPP_
#define FAST_LIVO2_ROS2_COMPAT_HPP_

#include <builtin_interfaces/msg/time.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/header.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace sensor_msgs
{
using Imu = msg::Imu;
using PointCloud2 = msg::PointCloud2;
using Image = msg::Image;
using ImageConstPtr = msg::Image::ConstSharedPtr;
namespace image_encodings = ::sensor_msgs::image_encodings;
} // namespace sensor_msgs

namespace nav_msgs
{
using Odometry = msg::Odometry;
using Path = msg::Path;
} // namespace nav_msgs

namespace geometry_msgs
{
using PoseStamped = msg::PoseStamped;
using Quaternion = msg::Quaternion;
} // namespace geometry_msgs

namespace visualization_msgs
{
using Marker = msg::Marker;
using MarkerArray = msg::MarkerArray;
} // namespace visualization_msgs

namespace fast_livo2
{
inline double stampToSec(const builtin_interfaces::msg::Time &stamp)
{
  return rclcpp::Time(stamp).seconds();
}

inline builtin_interfaces::msg::Time stampFromSec(double stamp)
{
  const auto nsec = static_cast<int64_t>(std::llround(stamp * 1e9));
  builtin_interfaces::msg::Time msg;
  msg.sec = static_cast<int32_t>(nsec / 1000000000LL);
  msg.nanosec = static_cast<uint32_t>(nsec % 1000000000LL);
  return msg;
}

inline builtin_interfaces::msg::Time timeToStamp(const rclcpp::Time &time)
{
  builtin_interfaces::msg::Time msg;
  msg.sec = static_cast<int32_t>(time.nanoseconds() / 1000000000LL);
  msg.nanosec = static_cast<uint32_t>(time.nanoseconds() % 1000000000LL);
  return msg;
}

inline builtin_interfaces::msg::Time nowStamp()
{
  const auto now = rclcpp::Clock(RCL_ROS_TIME).now();
  return timeToStamp(now);
}

inline builtin_interfaces::msg::Duration durationFromSec(double duration)
{
  const auto nsec = static_cast<int64_t>(std::llround(duration * 1e9));
  builtin_interfaces::msg::Duration msg;
  msg.sec = static_cast<int32_t>(nsec / 1000000000LL);
  msg.nanosec = static_cast<uint32_t>(nsec % 1000000000LL);
  return msg;
}

inline std::string ros1ParamNameToRos2(std::string name)
{
  for (char &c : name)
  {
    if (c == '/') c = '.';
  }
  return name;
}

template <typename T>
T declareOrGet(const rclcpp::Node::SharedPtr &node, const std::string &name, const T &default_value)
{
  const std::string ros2_name = ros1ParamNameToRos2(name);
  if (!node->has_parameter(ros2_name))
  {
    node->declare_parameter<T>(ros2_name, default_value);
  }
  T value = default_value;
  node->get_parameter(ros2_name, value);
  return value;
}

class Publisher
{
public:
  Publisher() = default;

  template <typename PublisherT, typename = std::enable_if_t<std::is_base_of_v<rclcpp::PublisherBase, PublisherT>>>
  explicit Publisher(std::shared_ptr<PublisherT> publisher)
    : base_(std::move(publisher))
  {
  }

  template <typename MessageT>
  void publish(const MessageT &msg) const
  {
    auto typed = std::dynamic_pointer_cast<rclcpp::Publisher<MessageT>>(base_);
    if (typed) typed->publish(msg);
  }

  template <typename MessageT>
  void publish(const std::shared_ptr<MessageT> &msg) const
  {
    if (msg) publish(*msg);
  }

  explicit operator bool() const { return static_cast<bool>(base_); }

private:
  std::shared_ptr<rclcpp::PublisherBase> base_;
};

} // namespace fast_livo2

#define ROS_INFO(...) RCLCPP_INFO(rclcpp::get_logger("fast_livo2"), __VA_ARGS__)
#define ROS_WARN(...) RCLCPP_WARN(rclcpp::get_logger("fast_livo2"), __VA_ARGS__)
#define ROS_ERROR(...) RCLCPP_ERROR(rclcpp::get_logger("fast_livo2"), __VA_ARGS__)
#define ROS_ASSERT(cond)                                                                            \
  do                                                                                                \
  {                                                                                                 \
    if (!(cond))                                                                                    \
    {                                                                                               \
      RCLCPP_FATAL(rclcpp::get_logger("fast_livo2"), "Assertion failed: %s", #cond);               \
      throw std::runtime_error("FAST-LIVO2 assertion failed: " #cond);                              \
    }                                                                                               \
  } while (0)

namespace tf
{
inline geometry_msgs::msg::Quaternion createQuaternionMsgFromRollPitchYaw(double roll, double pitch, double yaw)
{
  tf2::Quaternion q;
  q.setRPY(roll, pitch, yaw);
  return tf2::toMsg(q);
}
} // namespace tf

#endif // FAST_LIVO2_ROS2_COMPAT_HPP_

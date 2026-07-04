#ifndef FAST_LIVO2_VIKIT_CAMERA_LOADER_H_
#define FAST_LIVO2_VIKIT_CAMERA_LOADER_H_

#include <fast_livo2/ros2_compat.hpp>
#include <vikit/pinhole_camera.h>

#include <algorithm>
#include <memory>
#include <string>

namespace vk::camera_loader
{
inline bool loadFromRosNs(const rclcpp::Node::SharedPtr &node, vk::AbstractCamera *&cam)
{
  const auto get_string = [&](const std::string &name, const std::string &fallback) {
    return fast_livo2::declareOrGet<std::string>(node, "camera/" + name, fallback);
  };
  const auto get_int = [&](const std::string &name, int fallback) {
    return fast_livo2::declareOrGet<int>(node, "camera/" + name, fallback);
  };
  const auto get_double = [&](const std::string &name, double fallback) {
    return fast_livo2::declareOrGet<double>(node, "camera/" + name, fallback);
  };

  std::string model = get_string("cam_model", "RadTan");
  std::transform(model.begin(), model.end(), model.begin(), [](unsigned char c) { return std::tolower(c); });

  const int width = get_int("cam_width", 640);
  const int height = get_int("cam_height", 480);
  const double scale = get_double("scale", 1.0);
  const double fx = get_double("cam_fx", 0.0);
  const double fy = get_double("cam_fy", 0.0);
  const double cx = get_double("cam_cx", 0.0);
  const double cy = get_double("cam_cy", 0.0);
  if (width <= 0 || height <= 0 || fx <= 0.0 || fy <= 0.0)
  {
    RCLCPP_ERROR(node->get_logger(), "Invalid camera parameters: width=%d height=%d fx=%f fy=%f", width, height, fx, fy);
    return false;
  }

  delete cam;
  cam = nullptr;

  if (model == "equidistantcamera" || model == "equidistant" || model == "fisheye")
  {
    cam = new vk::EquidistantCamera(width, height, fx, fy, cx, cy,
                                    get_double("k1", 0.0), get_double("k2", 0.0),
                                    get_double("k3", 0.0), get_double("k4", 0.0), scale);
  }
  else
  {
    cam = new vk::RadTanCamera(width, height, fx, fy, cx, cy,
                               get_double("cam_d0", 0.0), get_double("cam_d1", 0.0),
                               get_double("cam_d2", 0.0), get_double("cam_d3", 0.0),
                               get_double("cam_d4", 0.0), scale);
  }
  RCLCPP_INFO(node->get_logger(), "Loaded camera model '%s' (%dx%d, scale %.3f)", model.c_str(), cam->width(), cam->height(), cam->scale());
  return true;
}
} // namespace vk::camera_loader

#endif // FAST_LIVO2_VIKIT_CAMERA_LOADER_H_


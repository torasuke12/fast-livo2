#include "LIVMapper.h"

#include <cstdlib>

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  options.allow_undeclared_parameters(true);
  options.automatically_declare_parameters_from_overrides(true);

  auto node = std::make_shared<rclcpp::Node>("laserMapping", options);
  if (std::getenv("FAST_LIVO2_DETERMINISTIC") != nullptr)
  {
    cv::setNumThreads(1);
    Eigen::setNbThreads(1);
    RCLCPP_INFO(node->get_logger(), "Deterministic processing enabled (single-threaded OpenCV/Eigen/OpenMP)");
  }
  LIVMapper mapper(node);
  mapper.initializeSubscribersAndPublishers();
  mapper.run();
  rclcpp::shutdown();
  return 0;
}

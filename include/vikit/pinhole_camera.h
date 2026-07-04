#ifndef FAST_LIVO2_VIKIT_PINHOLE_CAMERA_H_
#define FAST_LIVO2_VIKIT_PINHOLE_CAMERA_H_

#include <vikit/abstract_camera.h>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

namespace vk
{
class PinholeCamera : public AbstractCamera
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  PinholeCamera(int width, int height, double fx, double fy, double cx, double cy,
                double d0 = 0.0, double d1 = 0.0, double d2 = 0.0, double d3 = 0.0,
                double d4 = 0.0, double scale = 1.0)
  {
    width_ = static_cast<int>(width * scale);
    height_ = static_cast<int>(height * scale);
    scale_ = scale;
    fx_ = fx * scale;
    fy_ = fy * scale;
    cx_ = cx * scale;
    cy_ = cy * scale;
    dist_ = (cv::Mat_<double>(1, 5) << d0, d1, d2, d3, d4);
    K_ = (cv::Mat_<double>(3, 3) << fx_, 0.0, cx_, 0.0, fy_, cy_, 0.0, 0.0, 1.0);
  }

  Eigen::Vector3d cam2world(double u, double v) const override
  {
    std::vector<cv::Point2f> src{cv::Point2f(static_cast<float>(u), static_cast<float>(v))};
    std::vector<cv::Point2f> dst;
    cv::undistortPoints(src, dst, K_, dist_);
    Eigen::Vector3d ray(dst[0].x, dst[0].y, 1.0);
    return ray.normalized();
  }

  Eigen::Vector2d world2cam(const Eigen::Vector3d &xyz_c) const override
  {
    const double x = xyz_c.x() / xyz_c.z();
    const double y = xyz_c.y() / xyz_c.z();
    const double r2 = x * x + y * y;
    const double r4 = r2 * r2;
    const double r6 = r4 * r2;
    const double k1 = dist_.at<double>(0, 0);
    const double k2 = dist_.at<double>(0, 1);
    const double p1 = dist_.at<double>(0, 2);
    const double p2 = dist_.at<double>(0, 3);
    const double k3 = dist_.at<double>(0, 4);
    const double radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
    const double xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
    const double yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
    return Eigen::Vector2d(fx_ * xd + cx_, fy_ * yd + cy_);
  }

  void undistortImage(const cv::Mat &src, cv::Mat &dst) const
  {
    cv::undistort(src, dst, K_, dist_);
  }

private:
  cv::Mat K_;
  cv::Mat dist_;
};

class RadTanCamera : public PinholeCamera
{
public:
  using PinholeCamera::PinholeCamera;
};

class EquidistantCamera : public AbstractCamera
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EquidistantCamera(int width, int height, double fx, double fy, double cx, double cy,
                    double k1 = 0.0, double k2 = 0.0, double k3 = 0.0, double k4 = 0.0,
                    double scale = 1.0)
  {
    width_ = static_cast<int>(width * scale);
    height_ = static_cast<int>(height * scale);
    scale_ = scale;
    fx_ = fx * scale;
    fy_ = fy * scale;
    cx_ = cx * scale;
    cy_ = cy * scale;
    dist_ = (cv::Mat_<double>(1, 4) << k1, k2, k3, k4);
    K_ = (cv::Mat_<double>(3, 3) << fx_, 0.0, cx_, 0.0, fy_, cy_, 0.0, 0.0, 1.0);
  }

  Eigen::Vector3d cam2world(double u, double v) const override
  {
    std::vector<cv::Point2f> src{cv::Point2f(static_cast<float>(u), static_cast<float>(v))};
    std::vector<cv::Point2f> dst;
    cv::fisheye::undistortPoints(src, dst, K_, dist_);
    Eigen::Vector3d ray(dst[0].x, dst[0].y, 1.0);
    return ray.normalized();
  }

  Eigen::Vector2d world2cam(const Eigen::Vector3d &xyz_c) const override
  {
    std::vector<cv::Point3f> obj{cv::Point3f(static_cast<float>(xyz_c.x() / xyz_c.z()),
                                            static_cast<float>(xyz_c.y() / xyz_c.z()),
                                            1.0f)};
    std::vector<cv::Point2f> img;
    cv::fisheye::projectPoints(obj, img, cv::Vec3d::all(0.0), cv::Vec3d::all(0.0), K_, dist_);
    return Eigen::Vector2d(img[0].x, img[0].y);
  }

private:
  cv::Mat K_;
  cv::Mat dist_;
};
} // namespace vk

#endif // FAST_LIVO2_VIKIT_PINHOLE_CAMERA_H_


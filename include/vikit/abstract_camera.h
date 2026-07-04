#ifndef FAST_LIVO2_VIKIT_ABSTRACT_CAMERA_H_
#define FAST_LIVO2_VIKIT_ABSTRACT_CAMERA_H_

#include <Eigen/Core>

namespace vk
{
class AbstractCamera
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  virtual ~AbstractCamera() = default;

  virtual Eigen::Vector3d cam2world(double u, double v) const = 0;
  virtual Eigen::Vector3d cam2world(const Eigen::Vector2d &px) const { return cam2world(px.x(), px.y()); }
  virtual Eigen::Vector2d world2cam(const Eigen::Vector3d &xyz_c) const = 0;

  bool isInFrame(const Eigen::Vector2i &px, int boundary = 0) const
  {
    return px.x() >= boundary && px.y() >= boundary &&
           px.x() < width_ - boundary && px.y() < height_ - boundary;
  }

  int width() const { return width_; }
  int height() const { return height_; }
  double fx() const { return fx_; }
  double fy() const { return fy_; }
  double cx() const { return cx_; }
  double cy() const { return cy_; }
  double scale() const { return scale_; }

protected:
  int width_ = 0;
  int height_ = 0;
  double fx_ = 0.0;
  double fy_ = 0.0;
  double cx_ = 0.0;
  double cy_ = 0.0;
  double scale_ = 1.0;
};
} // namespace vk

#endif // FAST_LIVO2_VIKIT_ABSTRACT_CAMERA_H_


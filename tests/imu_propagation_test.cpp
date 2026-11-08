#include "IMU_Processing.h"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace
{
sensor_msgs::Imu::SharedPtr stationaryImu(const int64_t stamp_ns)
{
  auto imu = std::make_shared<sensor_msgs::Imu>();
  imu->header.stamp.sec = static_cast<int32_t>(stamp_ns / 1000000000LL);
  imu->header.stamp.nanosec = static_cast<uint32_t>(stamp_ns % 1000000000LL);
  imu->linear_acceleration.z = 1.0; // MID360 data uses g; Process2 estimates the scale.
  imu->angular_velocity.z = 0.01;  // Removed by stationary initialization.
  return imu;
}

void require(const bool condition, const char *message)
{
  if (!condition) throw std::runtime_error(message);
}

void requireFinite(const StatesGroup &state)
{
  require(state.rot_end.allFinite() && state.pos_end.allFinite() && state.vel_end.allFinite() &&
              state.cov.allFinite(),
          "IMU propagation produced a non-finite state or covariance");
}
}

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  int result = 0;
  try
  {
    ImuProcess imu;
    imu.first_lidar_time = 100.0;
    imu.set_imu_init_duration(3.0);
    imu.set_imu_init_frame_num(10);

    StatesGroup state;
    LidarMeasureGroup group;
    group.lidar.reset(new PointCloudXYZI());
    group.lio_vio_flg = LIO;
    MeasureGroup init;
    // The target image/scan time is 2 ms after the last IMU sample.
    init.lio_time = 103.002;
    for (int i = 0; i <= 600; ++i)
      init.imu.push_back(stationaryImu(100000000000LL + i * 5000000LL));
    group.measures.push_back(init);
    PointCloudXYZI::Ptr cloud(new PointCloudXYZI());
    imu.Process2(group, state, cloud);
    require(!imu.imu_need_init, "Stationary initialization did not complete");

    // LIVO immediately follows LIO with an image update at the same timestamp
    // and an empty IMU deque. It must not re-integrate the previous 2 ms tail.
    group.lio_vio_flg = VIO;
    group.measures.clear();
    MeasureGroup image;
    image.vio_time = init.lio_time;
    group.measures.push_back(image);
    const StatesGroup before(state);
    for (int i = 0; i < 3; ++i)
    {
      imu.Process2(group, state, cloud);
      requireFinite(state);
      require((state - before).norm() < 1e-12, "Same-time VIO update changed the state");
      require((state.cov - before.cov).norm() < 1e-12, "Same-time VIO update changed the covariance");
    }

    // If time genuinely advances without a new IMU sample, hold the last
    // measurement and integrate only from the previous propagation boundary.
    state.vel_end = V3D(1.0, -0.5, 0.25);
    const V3D velocity = state.vel_end;
    const V3D position = state.pos_end;
    group.measures.back().vio_time = 103.012;
    imu.Process2(group, state, cloud);
    requireFinite(state);
    require((state.vel_end - velocity).norm() < 1e-10, "Held stationary IMU changed velocity");
    require((state.pos_end - position - velocity * 0.01).norm() < 1e-10,
            "Empty-IMU update integrated an already propagated interval");

    // The ordinary scan path with new IMU samples must still integrate to the
    // exact scan boundary, including its final 2 ms extrapolation.
    group.lio_vio_flg = LIO;
    group.measures.clear();
    MeasureGroup scan;
    scan.lio_time = 103.022;
    scan.imu.push_back(stationaryImu(103015000000LL));
    scan.imu.push_back(stationaryImu(103020000000LL));
    group.measures.push_back(scan);
    imu.Process2(group, state, cloud);
    requireFinite(state);
    require((state.pos_end - position - velocity * 0.02).norm() < 1e-10,
            "Normal LIO propagation did not integrate the expected interval");
    std::cout << "IMU empty-sample and same-time propagation regression passed" << std::endl;
  }
  catch (const std::exception &error)
  {
    std::cerr << error.what() << std::endl;
    result = 1;
  }
  rclcpp::shutdown();
  return result;
}

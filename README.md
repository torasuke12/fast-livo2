# fast-livo2

ROS 2 Humble port of FAST-LIVO2 for the Unitree Go2W integration workspace.

This package is intended to be used as the `humble_ws/src/fast-livo2` submodule of `fast-livo2-go2w`. The runtime path is adapted for:

- Hesai XT16 LiDAR published as `sensor_msgs/msg/PointCloud2`
- Go2W body IMU published as `sensor_msgs/msg/Imu`
- Intel RealSense color camera images from `realsense-ros`

The upstream FAST-LIVO2 source was ROS Noetic based and targeted a different sensor stack. This package keeps the FAST-LIVO2 algorithm structure while adding ROS 2 Humble node, parameter, launch, message, and sensor integration code for Go2W.

## Status

- ROS distribution: Humble
- Package name: `fast_livo2`
- Main executable: `fastlivo2_mapping`
- Primary config: `config/go2w_fast_livo2.yaml`
- Primary launch: `launch/go2w_fast_livo2.launch.py`
- Sensor bringup launch: `launch/go2w_sensors.launch.py`

The package builds in the Docker workspace provided by `fast-livo2-go2w`. The default calibration file contains placeholders and must not be used for evaluation or field operation without replacement.

## Expected Topics

| Stream | Topic | Type | Use |
| --- | --- | --- | --- |
| LiDAR | `/hesai_node/points_raw` | `sensor_msgs/msg/PointCloud2` | LIO point cloud input |
| IMU | `/go2w/imu` | `sensor_msgs/msg/Imu` | Go2W body IMU input |
| Camera image | `/camera/camera/color/image_raw` | `sensor_msgs/msg/Image` | Visual update input |
| Camera info | `/camera/camera/color/camera_info` | `sensor_msgs/msg/CameraInfo` | Calibration source, not directly subscribed by FAST-LIVO2 |

The topic names are configured in `config/go2w_fast_livo2.yaml`.

## Repository Layout

```text
fast-livo2/
  CMakeLists.txt
  package.xml
  config/
    go2w_fast_livo2.yaml
    ...
  include/
  launch/
    go2w_fast_livo2.launch.py
    go2w_sensors.launch.py
  rviz_cfg/
  scripts/
  src/
```

## Use in fast-livo2-go2w

The recommended public workspace entry point is the parent repository:

```bash
git clone --recurse-submodules https://github.com/torasuke12/fast-livo2-go2w.git
cd fast-livo2-go2w
bash docker/robot/build.sh
bash docker/robot/run.sh
```

Inside the container:

```bash
cd /external
./scripts/build_humble_ws.sh
```

Run FAST-LIVO2 with sensor bringup:

```bash
source humble_ws/install/setup.bash
ros2 launch fast_livo2 go2w_fast_livo2.launch.py
```

Run FAST-LIVO2 only when LiDAR, IMU, and camera topics are already available:

```bash
ros2 launch fast_livo2 go2w_fast_livo2.launch.py use_sensor_bringup:=false
```

Run only the sensor bringup defined by this package:

```bash
ros2 launch fast_livo2 go2w_sensors.launch.py
```

## Standalone Build

This package can also be placed in a normal ROS 2 Humble workspace, but the required sensor packages must be available in the same workspace or installed environment:

- `go2w_imu_publisher`
- `hesai_lidar`
- `realsense2_camera`
- `unitree_go` and `unitree_api` dependencies used by the Go2W IMU bridge

Example:

```bash
mkdir -p ~/humble_ws/src
cd ~/humble_ws/src
git clone https://github.com/torasuke12/fast-livo2.git fast-livo2

cd ~/humble_ws
rosdep install -y -r --from-paths src --ignore-src --rosdistro humble
colcon build --symlink-install --packages-select fast_livo2 --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

For Go2W development, use the parent `fast-livo2-go2w` workspace instead of assembling these dependencies manually.

## Launch Arguments

`go2w_fast_livo2.launch.py`:

| Argument | Default | Description |
| --- | --- | --- |
| `params_file` | `config/go2w_fast_livo2.yaml` | FAST-LIVO2 parameter file |
| `log_level` | `info` | ROS logger level for `fastlivo2_mapping` |
| `use_sim_time` | `false` | Use `/clock` for bag reconstruction |
| `use_sensor_bringup` | `true` | Include `go2w_sensors.launch.py` |
| `use_hesai` | `true` | Start Hesai LiDAR driver launch |
| `use_go2w_imu` | `true` | Start Go2W IMU publisher |
| `use_realsense` | `true` | Start RealSense camera launch |
| `color_profile` | `640x480x30` | RealSense color stream profile |
| `enable_depth` | `false` | Enable RealSense depth stream |
| `enable_rs_imu` | `false` | Enable RealSense IMU streams |

Pass any RealSense selection argument when needed:

```bash
ros2 launch fast_livo2 go2w_fast_livo2.launch.py \
  serial_no:=_123456789 \
  color_profile:=1280x720x30
```

Use a calibrated parameter file:

```bash
ros2 launch fast_livo2 go2w_fast_livo2.launch.py \
  params_file:=/path/to/go2w_fast_livo2_calibrated.yaml
```

## Calibration

`config/go2w_fast_livo2.yaml` contains bootstrapping values only. Replace at least these fields before meaningful odometry use:

- `camera.cam_fx`, `camera.cam_fy`, `camera.cam_cx`, `camera.cam_cy`
- `camera.cam_d0` through `camera.cam_d4`
- `extrin_calib.extrinsic_T`
- `extrin_calib.extrinsic_R`
- `extrin_calib.Pcl`
- `extrin_calib.Rcl`
- `time_offset.imu_time_offset`
- `time_offset.img_time_offset`
- `time_offset.lidar_time_offset`

Frame interpretation:

- `extrinsic_T` and `extrinsic_R`: LiDAR to Go2W IMU transform used by the LIO state.
- `Pcl` and `Rcl`: LiDAR to RealSense color camera transform used by visual projection.
- Camera intrinsics should be taken from `/camera/camera/color/camera_info` for the exact RealSense model, serial number, stream profile, and resolution used at runtime.

The default identity extrinsics are placeholders, not a valid Go2W calibration.

## ROS 2 Port Notes

This fork adds or adapts:

- `rclcpp` node setup, parameters, publishers, subscribers, and timers
- ROS 2 `sensor_msgs`, `nav_msgs`, `geometry_msgs`, and `visualization_msgs` usage
- `tf2_ros` transform publishing
- Hesai `PointCloud2` preprocessing path for the Go2W XT16 setup
- Go2W body IMU input instead of assuming a LiDAR-integrated IMU
- Minimal vikit-compatible camera utility headers used by FAST-LIVO2
- RealSense-compatible camera model parameter loading

Livox compatibility structs are kept only so the original preprocessing code can compile. The supported Go2W runtime path uses Hesai `PointCloud2`.

## Known Limitations

- Online calibration is not implemented.
- The default config is not calibrated for any physical Go2W platform.
- Time synchronization between Go2W IMU, Hesai LiDAR, and RealSense camera must be validated on the target robot.
- The primary supported path is Go2W + Hesai XT16 + RealSense. Other upstream launch files and sensor modes are preserved where practical but are not the main tested deployment path.

## Upstream Attribution

This repository is derived from FAST-LIVO2:

- Upstream project: `hku-mars/FAST-LIVO2`
- Paper: FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry
- Original developer: Chunran Zheng

Keep upstream attribution and licensing when redistributing this package.

## License

FAST-LIVO2 is released under GPLv2. This ROS 2 Humble port keeps the same license declaration in `package.xml` as `GPL-2.0-only`. See `LICENSE` for the full license text.

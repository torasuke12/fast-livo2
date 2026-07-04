from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_hesai = LaunchConfiguration("use_hesai")
    use_go2w_imu = LaunchConfiguration("use_go2w_imu")
    use_realsense = LaunchConfiguration("use_realsense")
    enable_depth = LaunchConfiguration("enable_depth")
    enable_rs_imu = LaunchConfiguration("enable_rs_imu")

    hesai_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("hesai_lidar"), "launch", "hesai_lidar_launch.py"]
            )
        ),
        condition=IfCondition(use_hesai),
    )

    go2w_imu_node = Node(
        package="go2w_imu_publisher",
        executable="imu_publisher",
        name="go2w_imu_publisher",
        output="screen",
        condition=IfCondition(use_go2w_imu),
    )

    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("realsense2_camera"), "launch", "rs_launch.py"]
            )
        ),
        launch_arguments={
            "camera_namespace": LaunchConfiguration("camera_namespace"),
            "camera_name": LaunchConfiguration("camera_name"),
            "serial_no": LaunchConfiguration("serial_no"),
            "usb_port_id": LaunchConfiguration("usb_port_id"),
            "device_type": LaunchConfiguration("device_type"),
            "initial_reset": LaunchConfiguration("initial_reset"),
            "log_level": LaunchConfiguration("realsense_log_level"),
            "enable_color": "true",
            "rgb_camera.color_profile": LaunchConfiguration("color_profile"),
            "rgb_camera.color_format": LaunchConfiguration("color_format"),
            "enable_depth": enable_depth,
            "depth_module.depth_profile": LaunchConfiguration("depth_profile"),
            "enable_sync": LaunchConfiguration("enable_sync"),
            "enable_gyro": enable_rs_imu,
            "enable_accel": enable_rs_imu,
            "unite_imu_method": LaunchConfiguration("unite_rs_imu_method"),
            "publish_tf": "true",
            "tf_publish_rate": "0.0",
            "pointcloud.enable": "false",
            "align_depth.enable": "false",
        }.items(),
        condition=IfCondition(use_realsense),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_hesai", default_value="true"),
            DeclareLaunchArgument("use_go2w_imu", default_value="true"),
            DeclareLaunchArgument("use_realsense", default_value="true"),
            DeclareLaunchArgument("camera_namespace", default_value="camera"),
            DeclareLaunchArgument("camera_name", default_value="camera"),
            DeclareLaunchArgument("serial_no", default_value="''"),
            DeclareLaunchArgument("usb_port_id", default_value="''"),
            DeclareLaunchArgument("device_type", default_value="''"),
            DeclareLaunchArgument("initial_reset", default_value="false"),
            DeclareLaunchArgument("realsense_log_level", default_value="info"),
            DeclareLaunchArgument("color_profile", default_value="640x480x30"),
            DeclareLaunchArgument("color_format", default_value="RGB8"),
            DeclareLaunchArgument("enable_depth", default_value="false"),
            DeclareLaunchArgument("depth_profile", default_value="640x480x30"),
            DeclareLaunchArgument("enable_sync", default_value="true"),
            DeclareLaunchArgument("enable_rs_imu", default_value="false"),
            DeclareLaunchArgument("unite_rs_imu_method", default_value="0"),
            hesai_launch,
            go2w_imu_node,
            realsense_launch,
        ]
    )

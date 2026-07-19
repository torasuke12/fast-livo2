from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_livox = LaunchConfiguration("use_livox")
    use_hesai = LaunchConfiguration("use_hesai")
    use_go2w_imu = LaunchConfiguration("use_go2w_imu")
    use_realsense = LaunchConfiguration("use_realsense")
    enable_depth = LaunchConfiguration("enable_depth")
    enable_rs_imu = LaunchConfiguration("enable_rs_imu")

    livox_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("fast_livo2"), "launch", "go2w_livox_mid360.launch.py"]
            )
        ),
        launch_arguments={
            "xfer_format": LaunchConfiguration("livox_xfer_format"),
            "multi_topic": LaunchConfiguration("livox_multi_topic"),
            "data_src": LaunchConfiguration("livox_data_src"),
            "publish_freq": LaunchConfiguration("livox_publish_freq"),
            "output_data_type": LaunchConfiguration("livox_output_data_type"),
            "frame_id": LaunchConfiguration("livox_frame_id"),
            "user_config_path": LaunchConfiguration("livox_config"),
            "cmdline_input_bd_code": LaunchConfiguration("livox_bd_code"),
        }.items(),
        condition=IfCondition(use_livox),
    )

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
            DeclareLaunchArgument("use_livox", default_value="true"),
            DeclareLaunchArgument("use_hesai", default_value="false"),
            DeclareLaunchArgument("use_go2w_imu", default_value="false"),
            DeclareLaunchArgument("use_realsense", default_value="true"),
            DeclareLaunchArgument(
                "livox_config",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("fast_livo2"), "config", "MID360_config.json"]
                ),
            ),
            DeclareLaunchArgument("livox_xfer_format", default_value="1"),
            DeclareLaunchArgument("livox_multi_topic", default_value="0"),
            DeclareLaunchArgument("livox_data_src", default_value="0"),
            DeclareLaunchArgument("livox_publish_freq", default_value="10.0"),
            DeclareLaunchArgument("livox_output_data_type", default_value="0"),
            DeclareLaunchArgument("livox_frame_id", default_value="livox_frame"),
            DeclareLaunchArgument("livox_bd_code", default_value="livox0000000001"),
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
            livox_launch,
            hesai_launch,
            go2w_imu_node,
            realsense_launch,
        ]
    )

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    params_file = LaunchConfiguration("params_file")
    use_sensor_bringup = LaunchConfiguration("use_sensor_bringup")

    sensor_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("fast_livo2"),
                    "launch",
                    "go2w_sensors.launch.py",
                ]
            )
        ),
        launch_arguments={
            "use_hesai": LaunchConfiguration("use_hesai"),
            "use_go2w_imu": LaunchConfiguration("use_go2w_imu"),
            "use_realsense": LaunchConfiguration("use_realsense"),
            "camera_namespace": LaunchConfiguration("camera_namespace"),
            "camera_name": LaunchConfiguration("camera_name"),
            "serial_no": LaunchConfiguration("serial_no"),
            "usb_port_id": LaunchConfiguration("usb_port_id"),
            "device_type": LaunchConfiguration("device_type"),
            "initial_reset": LaunchConfiguration("initial_reset"),
            "realsense_log_level": LaunchConfiguration("realsense_log_level"),
            "color_profile": LaunchConfiguration("color_profile"),
            "color_format": LaunchConfiguration("color_format"),
            "enable_depth": LaunchConfiguration("enable_depth"),
            "depth_profile": LaunchConfiguration("depth_profile"),
            "enable_sync": LaunchConfiguration("enable_sync"),
            "enable_rs_imu": LaunchConfiguration("enable_rs_imu"),
            "unite_rs_imu_method": LaunchConfiguration("unite_rs_imu_method"),
        }.items(),
        condition=IfCondition(use_sensor_bringup),
    )

    fast_livo2_node = Node(
        package="fast_livo2",
        executable="fastlivo2_mapping",
        name="laserMapping",
        output="screen",
        parameters=[
            params_file,
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
        ],
        arguments=["--ros-args", "--log-level", LaunchConfiguration("log_level")],
    )

    default_params = PathJoinSubstitution(
        [FindPackageShare("fast_livo2"), "config", "go2w_fast_livo2.yaml"]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("params_file", default_value=default_params),
            DeclareLaunchArgument("log_level", default_value="info"),
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            DeclareLaunchArgument("use_sensor_bringup", default_value="true"),
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
            sensor_bringup,
            fast_livo2_node,
        ]
    )

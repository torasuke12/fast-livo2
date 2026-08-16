from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
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
            "use_livox": LaunchConfiguration("use_livox"),
            "use_hesai": LaunchConfiguration("use_hesai"),
            "use_go2w_imu": LaunchConfiguration("use_go2w_imu"),
            "use_realsense": LaunchConfiguration("use_realsense"),
            "livox_config": LaunchConfiguration("livox_config"),
            "livox_xfer_format": LaunchConfiguration("livox_xfer_format"),
            "livox_multi_topic": LaunchConfiguration("livox_multi_topic"),
            "livox_data_src": LaunchConfiguration("livox_data_src"),
            "livox_publish_freq": LaunchConfiguration("livox_publish_freq"),
            "livox_output_data_type": LaunchConfiguration("livox_output_data_type"),
            "livox_frame_id": LaunchConfiguration("livox_frame_id"),
            "livox_bd_code": LaunchConfiguration("livox_bd_code"),
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
            {
                "use_sim_time": LaunchConfiguration("use_sim_time"),
                "common.reliable_sensor_qos": ParameterValue(
                    LaunchConfiguration("reliable_sensor_qos"), value_type=bool
                ),
                "common.img_en": ParameterValue(
                    LaunchConfiguration("img_en"), value_type=int
                ),
            },
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
            DeclareLaunchArgument("reliable_sensor_qos", default_value="false"),
            DeclareLaunchArgument(
                "img_en",
                default_value="1",
                description="Set to 0 to disable VIO and run LIO-only mode",
            ),
            DeclareLaunchArgument("use_sensor_bringup", default_value="true"),
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
            DeclareLaunchArgument("livox_xfer_format", default_value="0"),
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
            sensor_bringup,
            fast_livo2_node,
        ]
    )

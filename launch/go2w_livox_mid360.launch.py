from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_config = PathJoinSubstitution(
        [FindPackageShare("fast_livo2"), "config", "MID360_config.json"]
    )

    livox_node = Node(
        package="livox_ros_driver2",
        executable="livox_ros_driver2_node",
        name="livox_lidar_publisher",
        output="screen",
        parameters=[
            {
                "xfer_format": ParameterValue(LaunchConfiguration("xfer_format"), value_type=int),
                "multi_topic": ParameterValue(LaunchConfiguration("multi_topic"), value_type=int),
                "data_src": ParameterValue(LaunchConfiguration("data_src"), value_type=int),
                "publish_freq": ParameterValue(LaunchConfiguration("publish_freq"), value_type=float),
                "output_data_type": ParameterValue(LaunchConfiguration("output_data_type"), value_type=int),
                "frame_id": LaunchConfiguration("frame_id"),
                "user_config_path": LaunchConfiguration("user_config_path"),
                "cmdline_input_bd_code": LaunchConfiguration("cmdline_input_bd_code"),
            }
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("xfer_format", default_value="0"),
            DeclareLaunchArgument("multi_topic", default_value="0"),
            DeclareLaunchArgument("data_src", default_value="0"),
            DeclareLaunchArgument("publish_freq", default_value="10.0"),
            DeclareLaunchArgument("output_data_type", default_value="0"),
            DeclareLaunchArgument("frame_id", default_value="livox_frame"),
            DeclareLaunchArgument("user_config_path", default_value=default_config),
            DeclareLaunchArgument("cmdline_input_bd_code", default_value="livox0000000001"),
            livox_node,
        ]
    )

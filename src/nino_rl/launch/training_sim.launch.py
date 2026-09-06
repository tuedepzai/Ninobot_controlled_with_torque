"""Start the Nino cable-bump world plus the episode-reset service bridge."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    simulator = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("nino_description"), "launch", "sim.launch.py"]
            )
        ),
        launch_arguments={
            "world": "long_hall.sdf",
            "world_name": "long_hall",
            "headless": LaunchConfiguration("headless"),
            "rviz": LaunchConfiguration("rviz"),
            "sensor_monitor": "false",
            "start_effort_drive": "true",
            "linorobot2_mode": "false",
            "verbosity": LaunchConfiguration("verbosity"),
        }.items(),
    )
    reset_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="rl_world_control_bridge",
        output="screen",
        arguments=[
            "/world/long_hall/control@ros_gz_interfaces/srv/ControlWorld",
            "/world/long_hall/set_pose@ros_gz_interfaces/srv/SetEntityPose",
        ],
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "headless", default_value="true", description="Disable Gazebo GUI while training"
            ),
            DeclareLaunchArgument(
                "rviz", default_value="false", description="Open RViz for debugging"
            ),
            DeclareLaunchArgument(
                "verbosity", default_value="1", description="Gazebo log level (0-4)"
            ),
            simulator,
            reset_bridge,
        ]
    )

"""Build a map from Nino's existing /scan, /odom and TF interfaces."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    share = FindPackageShare("nino_navigation")
    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("rviz", default_value="true"),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution([
                FindPackageShare("slam_toolbox"), "launch", "online_async_launch.py"
            ])),
            launch_arguments={
                "use_sim_time": LaunchConfiguration("use_sim_time"),
                "slam_params_file": PathJoinSubstitution([
                    share, "config", "slam.yaml"
                ]),
            }.items(),
        ),
        Node(
            package="rviz2", executable="rviz2", name="rviz2",
            arguments=["-d", PathJoinSubstitution([share, "rviz", "navigation.rviz"])],
            parameters=[{"use_sim_time": LaunchConfiguration("use_sim_time")}],
            condition=IfCondition(LaunchConfiguration("rviz")),
        ),
    ])

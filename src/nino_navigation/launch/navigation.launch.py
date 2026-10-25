"""Localize on a saved map and run Nav2 against Nino's torque adapter."""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def launch_navigation(context):
    map_path = Path(LaunchConfiguration("map").perform(context)).expanduser().resolve()
    if not map_path.is_file():
        raise RuntimeError(f"Saved map YAML does not exist: {map_path}")
    if LaunchConfiguration("fixed_path").perform(context).lower() == "true":
        return [IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution([
                FindPackageShare("nino_navigation"), "launch", "fixed_path.launch.py"
            ])),
            launch_arguments={
                "map": str(map_path),
                "use_sim_time": LaunchConfiguration("use_sim_time"),
                "params_file": LaunchConfiguration("params_file"),
                "max_path_deviation": LaunchConfiguration("max_path_deviation"),
            }.items(),
        )]
    return [IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare("nav2_bringup"), "launch", "bringup_launch.py"
        ])),
        launch_arguments={
            "map": str(map_path),
            "use_sim_time": LaunchConfiguration("use_sim_time"),
            "params_file": LaunchConfiguration("params_file"),
            "slam": "False",
            "autostart": "true",
            "use_composition": "False",
        }.items(),
    )]


def generate_launch_description():
    share = FindPackageShare("nino_navigation")
    return LaunchDescription([
        DeclareLaunchArgument("map", description="Path to a saved map YAML"),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("fixed_path", default_value="true",
                              choices=["true", "false"],
                              description="Plan once per episode; false restores standard Nav2 replanning"),
        DeclareLaunchArgument("max_path_deviation", default_value="0.20"),
        DeclareLaunchArgument("params_file", default_value=PathJoinSubstitution([
            share, "config", "nav2.yaml"
        ])),
        OpaqueFunction(function=launch_navigation),
        Node(
            package="rviz2", executable="rviz2", name="rviz2",
            arguments=["-d", PathJoinSubstitution([share, "rviz", "navigation.rviz"])],
            parameters=[{"use_sim_time": LaunchConfiguration("use_sim_time")}],
            condition=IfCondition(LaunchConfiguration("rviz")),
        ),
    ])

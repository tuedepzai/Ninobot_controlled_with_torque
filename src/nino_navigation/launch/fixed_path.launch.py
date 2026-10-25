"""Only the Nav2 servers needed for one-plan episodes; no BT/recovery clients."""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def bringup(context):
    map_path = Path(LaunchConfiguration('map').perform(context)).expanduser().resolve()
    if not map_path.is_file():
        raise RuntimeError(f'Saved map YAML does not exist: {map_path}')
    params = LaunchConfiguration('params_file')
    sim = {'use_sim_time': ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool)}
    actions = [IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare('nav2_bringup'), 'launch', 'localization_launch.py'])),
        launch_arguments={'map': str(map_path), 'params_file': params,
                          'use_sim_time': LaunchConfiguration('use_sim_time'),
                          'use_composition': 'False', 'autostart': 'true'}.items())]
    servers = [
        ('nav2_controller', 'controller_server', [('cmd_vel', 'cmd_vel_nav')], {}),
        ('nav2_planner', 'planner_server', [], {}),
        ('nav2_velocity_smoother', 'velocity_smoother', [('cmd_vel', 'cmd_vel_nav')], {}),
        ('nav2_collision_monitor', 'collision_monitor', [],
         {'cmd_vel_out_topic': '/episode/cmd_vel_candidate'}),
    ]
    for package, name, remaps, overrides in servers:
        actions.append(Node(package=package, executable=name, name=name, output='screen',
                            parameters=[params, sim, overrides], remappings=remaps))
    actions.extend([
        Node(package='nav2_lifecycle_manager', executable='lifecycle_manager',
             name='lifecycle_manager_navigation', output='screen',
             parameters=[sim, {'autostart': True, 'node_names': [s[1] for s in servers]}]),
        Node(package='nino_navigation', executable='episode_manager', name='episode_manager',
             output='screen', parameters=[PathJoinSubstitution([
                 FindPackageShare('nino_navigation'), 'config', 'episode.yaml']), sim,
                 {'max_path_deviation': ParameterValue(
                     LaunchConfiguration('max_path_deviation'), value_type=float)}]),
    ])
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('map'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('max_path_deviation', default_value='0.20'),
        DeclareLaunchArgument('params_file', default_value=PathJoinSubstitution([
            FindPackageShare('nino_navigation'), 'config', 'nav2.yaml'])),
        OpaqueFunction(function=bringup),
    ])

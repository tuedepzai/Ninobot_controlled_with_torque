"""Spawn Nino at world origin in Gazebo Sim and bridge its ROS interfaces."""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    RegisterEventHandler,
    SetEnvironmentVariable,
    Shutdown,
    UnsetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import (
    Command,
    EnvironmentVariable,
    IfElseSubstitution,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from launch_ros.substitutions import FindPackagePrefix


SNAP_DESKTOP_ENVIRONMENT = (
    "GDK_PIXBUF_MODULEDIR",
    "GDK_PIXBUF_MODULE_FILE",
    "GIO_LAUNCHED_DESKTOP_FILE",
    "GIO_MODULE_DIR",
    "GSETTINGS_SCHEMA_DIR",
    "GTK_EXE_PREFIX",
    "GTK_IM_MODULE_FILE",
    "GTK_MODULES",
    "GTK_PATH",
    "LOCPATH",
    "SNAP",
    "SNAP_ARCH",
    "SNAP_COMMON",
    "SNAP_CONTEXT",
    "SNAP_COOKIE",
    "SNAP_DATA",
    "SNAP_EUID",
    "SNAP_INSTANCE_NAME",
    "SNAP_LAUNCHER_ARCH_TRIPLET",
    "SNAP_LIBRARY_PATH",
    "SNAP_NAME",
    "SNAP_REAL_HOME",
    "SNAP_REVISION",
    "SNAP_UID",
    "SNAP_USER_COMMON",
    "SNAP_USER_DATA",
    "SNAP_VERSION",
    "XDG_DATA_HOME",
)
NVIDIA_EGL_VENDOR = Path("/usr/share/glvnd/egl_vendor.d/10_nvidia.json")


def generate_launch_description():
    package_share = FindPackageShare("nino_description")
    control_share = FindPackageShare("nino_control")
    world = PathJoinSubstitution(
        [package_share, "worlds", LaunchConfiguration("world")]
    )
    xacro_file = PathJoinSubstitution([package_share, "urdf", "nino.urdf.xacro"])
    effort_drive_config = PathJoinSubstitution(
        [control_share, "config", "effort_drive.yaml"]
    )
    robot_description = ParameterValue(Command(["xacro ", xacro_file]), value_type=str)

    clean_gz_sim = PathJoinSubstitution(
        [FindPackagePrefix("nino_description"), "lib", "nino_description", "gz_sim_clean"]
    )
    gazebo = ExecuteProcess(
        cmd=[
            clean_gz_sim,
            "-r",
            IfElseSubstitution(
                LaunchConfiguration("headless"), if_value="-s", else_value=""
            ),
            "-v",
            LaunchConfiguration("verbosity"),
            world,
            "--force-version",
            "8",
        ],
        name="gazebo",
        output="screen",
        on_exit=Shutdown(),
    )

    spawn_nino = Node(
        package="ros_gz_sim",
        executable="create",
        name="spawn_nino",
        output="screen",
        parameters=[{
            "world": LaunchConfiguration("world_name"),
            "name": "nino",
            "topic": "/robot_description",
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "Y": 0.0,
        }],
    )
    controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        name="control_spawner",
        output="screen",
        arguments=[
            "joint_state_broadcaster",
            "wheel_effort_controller",
            "--controller-manager",
            "/controller_manager",
            "--controller-manager-timeout",
            "30",
            "--switch-timeout",
            "30",
            "--activate-as-group",
        ],
    )
    effort_drive = Node(
        package="nino_control",
        executable="effort_drive",
        name="effort_drive",
        output="screen",
        condition=IfCondition(LaunchConfiguration("start_effort_drive")),
        parameters=[effort_drive_config, {
            "use_sim_time": True,
            "max_wheel_torque": ParameterValue(
                LaunchConfiguration("max_wheel_torque"), value_type=float
            ),
            "odom_topic": IfElseSubstitution(
                LaunchConfiguration("linorobot2_mode"),
                if_value="/odom/unfiltered",
                else_value="/odom",
            ),
            "publish_odom_tf": ParameterValue(
                IfElseSubstitution(
                    LaunchConfiguration("linorobot2_mode"),
                    if_value="false",
                    else_value="true",
                ),
                value_type=bool,
            ),
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "verbosity",
            default_value="2",
            description="Gazebo Sim console verbosity (0-4)",
        ),
        DeclareLaunchArgument(
            "headless",
            default_value="false",
            description="Run the Gazebo server without the graphical client",
        ),
        DeclareLaunchArgument(
            "world",
            default_value="long_hall.sdf",
            description="World file installed by nino_description",
        ),
        DeclareLaunchArgument(
            "world_name",
            default_value="long_hall",
            description="SDF world name used by the entity spawn service",
        ),
        DeclareLaunchArgument(
            "max_wheel_torque",
            default_value="12.0",
            description="Symmetric wheel torque limit in N.m (maximum 12.0)",
        ),
        DeclareLaunchArgument(
            "linorobot2_mode",
            default_value="false",
            description=(
                "Publish /odom/unfiltered without odom TF for Linorobot2's EKF"
            ),
        ),
        DeclareLaunchArgument(
            "start_effort_drive",
            default_value="true",
            description="Start the safe cmd_vel and torque adapter",
        ),
        SetEnvironmentVariable(
            "GZ_SIM_RESOURCE_PATH",
            [package_share, "/..:", EnvironmentVariable("GZ_SIM_RESOURCE_PATH", default_value="")],
        ),
        SetEnvironmentVariable(
            "GZ_SIM_SYSTEM_PLUGIN_PATH",
            [
                EnvironmentVariable("GZ_SIM_SYSTEM_PLUGIN_PATH", default_value=""),
                ":",
                EnvironmentVariable("LD_LIBRARY_PATH", default_value=""),
            ],
        ),
        # Keep Gazebo Transport discovery on the local loopback interface.
        # This is deterministic on hosts with VPN, Docker, or multiple NICs.
        SetEnvironmentVariable("GZ_IP", "127.0.0.1"),
        # VS Code installed as a Snap exports desktop library paths into its
        # integrated terminals. Gazebo GUI must use the host Ubuntu libraries.
        *[UnsetEnvironmentVariable(name) for name in SNAP_DESKTOP_ENVIRONMENT],
        SetEnvironmentVariable(
            "XDG_DATA_DIRS",
            "/usr/share/ubuntu:/usr/share/gnome:/usr/local/share:/usr/share:/var/lib/snapd/desktop",
        ),
        # Gazebo's bundled Qt 5 dialogs emit harmless binding-loop diagnostics.
        SetEnvironmentVariable("QT_LOGGING_RULES", "*.warning=false"),
        *(
            [
                SetEnvironmentVariable(
                    "__EGL_VENDOR_LIBRARY_FILENAMES",
                    str(NVIDIA_EGL_VENDOR),
                )
            ]
            if NVIDIA_EGL_VENDOR.exists()
            else []
        ),
        gazebo,
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            output="screen",
            parameters=[{"robot_description": robot_description, "use_sim_time": True}],
        ),
        spawn_nino,
        Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            name="ros_gz_bridge",
            output="screen",
            arguments=[
                "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            ],
        ),
        RegisterEventHandler(
            OnProcessExit(target_action=spawn_nino, on_exit=[controller_spawner])
        ),
        RegisterEventHandler(
            OnProcessExit(target_action=controller_spawner, on_exit=[effort_drive])
        ),
    ])

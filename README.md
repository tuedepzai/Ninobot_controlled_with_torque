# Nino Robot — ROS 2 Jazzy, Gazebo Sim, and torque control

This ROS 2 workspace contains the Nino differential-drive AMR description,
Gazebo Sim Harmonic world, and wheel-torque control stack. The robot spawns at
world pose `0 0 0` and can be driven through `/cmd_vel` or direct torque
commands.

The simulator uses this control path:

```text
/cmd_vel ──> effort_drive PI loop ──> /wheel_effort_controller/commands
                                             │
                                             v
                              JointGroupEffortController
                                             │
                                             v
                          left/right joint effort interfaces
                                             │
                                             v
                                  gz_ros2_control + Gazebo
```

## Workspace layout

```text
ninorobot/
├── README.md
└── src/
    ├── nino_description/
    │   ├── launch/sim.launch.py
    │   ├── meshes/
    │   ├── urdf/nino.urdf.xacro
    │   └── worlds/
    │       ├── long_hall.sdf
    │       └── flat_world.sdf
    └── nino_control/
        ├── config/controllers.yaml
        ├── config/effort_drive.yaml
        ├── nino_control/effort_drive.py
        ├── nino_control/kinematics.py
        └── test/test_kinematics.py
```

`nino_description` owns geometry and simulation bring-up. `nino_control` owns
controller configuration, torque control, differential-drive kinematics, and
wheel odometry.

## Install dependencies

ROS 2 Jazzy binary packages target Ubuntu 24.04. Ubuntu 22.04 requires a Jazzy
source build or an Ubuntu 24.04 ROS container.

```bash
sudo apt update
sudo apt install ros-jazzy-ros-gz ros-jazzy-xacro \
  ros-jazzy-robot-state-publisher ros-jazzy-teleop-twist-keyboard \
  ros-jazzy-gz-ros2-control ros-jazzy-controller-manager \
  ros-jazzy-effort-controllers \
  ros-jazzy-joint-state-broadcaster ros-jazzy-ros2controlcli \
  python3-colcon-common-extensions
```

You can also let `rosdep` resolve package dependencies. Initialize it once,
then install everything declared by this workspace:

```bash
sudo rosdep init
rosdep update
cd /home/tue/ninorobot
rosdep install --from-paths src --ignore-src --rosdistro jazzy -r -y
```

If `rosdep init` reports that its sources list already exists, skip that first
command and continue with `rosdep update`.

## Build and run

```bash
cd /home/tue/ninorobot
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
ros2 launch nino_description sim.launch.py
```

For a server-only run:

```bash
ros2 launch nino_description sim.launch.py headless:=true
```

The robot spawns at `x=0`, `y=0`, `z=0`, and `yaw=0`. Do not start a second
copy of the launch file while one is already running. `Ctrl-C` requests a
clean Gazebo server stop and should finish without a false process-crash error.

## Verify ros2_control

With the simulation running:

```bash
ros2 control list_controllers
ros2 control list_hardware_interfaces
ros2 topic hz /joint_states
```

The expected active controllers are `joint_state_broadcaster` and
`wheel_effort_controller`. The claimed command interfaces are:

```text
left_wheel_joint/effort
right_wheel_joint/effort
```

The controller's input is a `std_msgs/msg/Float64MultiArray` ordered as
`[left_wheel_joint, right_wheel_joint]`.

## Drive with `/cmd_vel`

The default `effort_drive` node converts desired base velocity into wheel-speed
targets, closes a PI loop using measured `/joint_states`, and sends bounded
torque to `JointGroupEffortController`.

Keyboard control:

```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r cmd_vel:=/cmd_vel
```

Constant forward command:

```bash
ros2 topic pub --rate 10 /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.3}, angular: {z: 0.0}}"
```

The `/cmd_vel` watchdog applies zero target velocity after 0.5 seconds without
a new command.

## Control wheel torque directly

The safe direct-torque input is `/wheel_torque_commands`. Values are N·m in
left/right order, are clamped to the configured limit, and expire after 0.25 s.

Apply `0.8 N·m` to both wheels:

```bash
ros2 topic pub --rate 20 /wheel_torque_commands \
  std_msgs/msg/Float64MultiArray "{data: [0.8, 0.8]}"
```

The two entries are independent. For example, apply `0.8 N·m` to the left
wheel and `0.4 N·m` to the right wheel:

```bash
ros2 topic pub --rate 20 /wheel_torque_commands \
  std_msgs/msg/Float64MultiArray "{data: [0.8, 0.4]}"
```

Use a negative value to reverse a wheel. The order is always
`[left_wheel_joint, right_wheel_joint]`.

Stop the publisher with `Ctrl-C`. The timeout returns both commands to zero.
Monitor the torque actually sent by the adapter:

```bash
ros2 topic echo /wheel_torque_applied
```

This topic reports the safe applied command, which can be lower than the
requested torque while the torque ramp or wheel-speed guard is active.

To inspect `JointGroupEffortController` without the safety adapter, launch with:

```bash
ros2 launch nino_description sim.launch.py start_effort_drive:=false
```

Then publish to the controller directly:

```bash
ros2 topic pub --rate 20 /wheel_effort_controller/commands \
  std_msgs/msg/Float64MultiArray "{data: [0.8, 0.8]}"
```

The raw interface bypasses the watchdog, torque ramp, and software speed guard;
use it only with the robot restrained or for a short controller check. It holds
its last received effort. Always send zero before stopping the raw publisher:

```bash
ros2 topic pub --once /wheel_effort_controller/commands \
  std_msgs/msg/Float64MultiArray "{data: [0.0, 0.0]}"
```

Limit the simulated wheel torque below the URDF maximum when launching:

```bash
ros2 launch nino_description sim.launch.py max_wheel_torque:=3.0
```

The hard URDF and ros2_control interface limit is `±12 N·m`. The default
`/cmd_vel` speed loop is separately limited to `±2 N·m`; direct torque mode can
use the full configured hardware limit. PI gains, acceleration limits, watchdog
times, and odometry settings are kept in
`src/nino_control/config/effort_drive.yaml`.

Direct effort is slew-limited to `10 N·m/s`. A `12 rad/s` software wheel-speed
guard removes torque that would accelerate an overspeed wheel, while allowing
opposite braking torque. The URDF's `24 rad/s` emergency limit remains enabled
as a final independent backstop.

## Nav2 and Linorobot2 compatibility

No Nav2 nodes are included or launched yet. The default simulation already
provides the mobile-base interfaces Nav2 needs:

- `/cmd_vel` — `geometry_msgs/msg/Twist`
- `/odom` — `nav_msgs/msg/Odometry`
- `odom -> base_footprint` TF
- `base_footprint -> base_link` and wheel TF from `robot_state_publisher`

Linorobot2's Jazzy EKF expects raw wheel odometry on `/odom/unfiltered` and
publishes filtered `/odom` plus the odometry TF. Use:

```bash
ros2 launch nino_description sim.launch.py linorobot2_mode:=true
```

In this mode Nino publishes `/odom/unfiltered` and does not publish
`odom -> base_footprint`, avoiding duplicate `/odom` publishers or TF sources
when Linorobot2's `ekf_filter_node` runs. Use the `jazzy` branch of
[Linorobot2](https://github.com/linorobot/linorobot2), select its 2WD base
configuration, and keep its standard `cmd_vel`, `odom`, `base_footprint`, and
`base_link` names. A lidar `/scan` and normally IMU `/imu/data` must be added
before running its navigation stack.

## Moving from Gazebo to Xiaomi CyberGear hardware

`gz_ros2_control/GazeboSimSystem` is simulation-only. On the physical robot,
replace that hardware plugin with a ros2_control `SystemInterface` that:

1. Converts `left_wheel_joint/effort` and `right_wheel_joint/effort` from N·m
   into CyberGear CAN torque commands.
2. Reports measured joint position, velocity, and effort in SI units.
3. Enforces the motor, gearbox, electrical, thermal, and emergency-stop limits
   for the actual installation.

The controller and high-level topic contract can remain unchanged, so
`/cmd_vel`, direct torque commands, odometry, Linorobot2, and later Nav2 do not
need to know whether the joint backend is Gazebo or CyberGear CAN hardware.

## Physics and collision model

The supplied CAD masses, centers of mass, and inertia tensors are retained.
Left/right inertial and contact properties are symmetric; the effective wheel
separation uses the measured tyre contact centers. Rolling surfaces use exact
measured cylinders for stable contact. The caster bracket collision uses its
CAD mesh in the same frame as the visual mesh, keeping the open fork clear.
The modeled moving mass is `4.6888672492 kg`; its aggregate center of mass in
`base_footprint` is approximately `[0.056018, 0.0, 0.127546] m`. Every inertia
tensor is positive definite and satisfies the rigid-body triangle conditions.

The default closed hall is 34 m long and 4 m wide. Its fourteen cable bumps
vary from 16–44 mm diameter and −22° to +24°. Every angled cable length is
`4.0 / cos(angle)`, so it reaches both inner wall faces.
# Ninobot_controlled_with_torque
# Ninobot_controlled_with_torque
# Ninobot_controlled_with_torque

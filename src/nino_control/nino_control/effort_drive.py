"""Closed-loop /cmd_vel to wheel-torque adapter for JointGroupEffortController."""

from math import cos, isfinite, sin

import rclpy
from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from tf2_ros import TransformBroadcaster

from nino_control.kinematics import (
    clamp,
    integrate_wheel_odometry,
    limit_effort_commands,
    wheel_angular_targets,
)


class EffortDrive(Node):
    """Track differential wheel speeds with bounded torque commands."""

    def __init__(self) -> None:
        super().__init__("effort_drive")

        self.declare_parameter("left_wheel_joint", "left_wheel_joint")
        self.declare_parameter("right_wheel_joint", "right_wheel_joint")
        self.declare_parameter("wheel_radius", 0.0625)
        self.declare_parameter("wheel_separation", 0.34273666)
        # The URDF hard limit is 24 rad/s.  Keep a margin so direct torque
        # cannot drive controller_manager into that hard limit.
        self.declare_parameter("max_wheel_speed", 12.0)
        self.declare_parameter("max_wheel_acceleration", 12.0)
        self.declare_parameter("max_wheel_torque", 12.0)
        self.declare_parameter("max_velocity_control_torque", 2.0)
        self.declare_parameter("max_effort_rate", 10.0)
        self.declare_parameter("velocity_kp", 0.30)
        self.declare_parameter("velocity_ki", 0.10)
        self.declare_parameter("integral_limit", 4.0)
        self.declare_parameter("command_timeout", 0.5)
        self.declare_parameter("torque_command_timeout", 0.25)
        self.declare_parameter("control_rate", 1000.0)
        self.declare_parameter("odom_publish_rate", 50.0)
        self.declare_parameter("torque_status_publish_rate", 50.0)
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("odom_frame_id", "odom")
        self.declare_parameter("base_frame_id", "base_footprint")
        self.declare_parameter("publish_odom_tf", True)

        self.left_joint = str(self.get_parameter("left_wheel_joint").value)
        self.right_joint = str(self.get_parameter("right_wheel_joint").value)
        self.wheel_radius = float(self.get_parameter("wheel_radius").value)
        self.wheel_separation = float(self.get_parameter("wheel_separation").value)
        self.max_wheel_speed = float(self.get_parameter("max_wheel_speed").value)
        self.max_wheel_acceleration = float(
            self.get_parameter("max_wheel_acceleration").value
        )
        requested_max_torque = float(self.get_parameter("max_wheel_torque").value)
        if not isfinite(requested_max_torque) or requested_max_torque <= 0.0:
            raise ValueError("max_wheel_torque must be finite and positive")
        self.max_torque = min(requested_max_torque, 12.0)
        requested_velocity_torque = float(
            self.get_parameter("max_velocity_control_torque").value
        )
        if not isfinite(requested_velocity_torque) or requested_velocity_torque <= 0.0:
            raise ValueError("max_velocity_control_torque must be finite and positive")
        self.max_velocity_torque = min(requested_velocity_torque, self.max_torque)
        self.max_effort_rate = float(self.get_parameter("max_effort_rate").value)
        self.kp = float(self.get_parameter("velocity_kp").value)
        self.ki = float(self.get_parameter("velocity_ki").value)
        self.integral_limit = float(self.get_parameter("integral_limit").value)
        self.command_timeout = float(self.get_parameter("command_timeout").value)
        self.torque_timeout = float(self.get_parameter("torque_command_timeout").value)
        self.control_rate = float(self.get_parameter("control_rate").value)
        self.odom_publish_rate = float(self.get_parameter("odom_publish_rate").value)
        self.torque_status_publish_rate = float(
            self.get_parameter("torque_status_publish_rate").value
        )
        self.odom_frame = str(self.get_parameter("odom_frame_id").value)
        self.base_frame = str(self.get_parameter("base_frame_id").value)
        self.publish_odom_tf = bool(self.get_parameter("publish_odom_tf").value)

        positive_parameters = {
            "wheel_radius": self.wheel_radius,
            "wheel_separation": self.wheel_separation,
            "max_wheel_speed": self.max_wheel_speed,
            "max_wheel_acceleration": self.max_wheel_acceleration,
            "max_wheel_torque": self.max_torque,
            "max_velocity_control_torque": self.max_velocity_torque,
            "max_effort_rate": self.max_effort_rate,
            "control_rate": self.control_rate,
            "odom_publish_rate": self.odom_publish_rate,
            "torque_status_publish_rate": self.torque_status_publish_rate,
        }
        for name, value in positive_parameters.items():
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive, got {value}")
        for name, value in {"velocity_kp": self.kp, "velocity_ki": self.ki}.items():
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative, got {value}")
        if requested_max_torque > 12.0:
            self.get_logger().info(
                "max_wheel_torque is capped at the URDF limit of 12.0 N.m"
            )

        self.effort_publisher = self.create_publisher(
            Float64MultiArray, "/wheel_effort_controller/commands", 10
        )
        self.applied_torque_publisher = self.create_publisher(
            Float64MultiArray, "/wheel_torque_applied", 10
        )
        self.odom_publisher = self.create_publisher(
            Odometry, str(self.get_parameter("odom_topic").value), 10
        )
        self.tf_broadcaster = TransformBroadcaster(self) if self.publish_odom_tf else None

        self.create_subscription(Twist, "/cmd_vel", self._cmd_vel_callback, 10)
        self.create_subscription(
            Float64MultiArray,
            "/wheel_torque_commands",
            self._torque_callback,
            10,
        )
        wheel_state_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.create_subscription(
            JointState, "/joint_states", self._joint_state_callback, wheel_state_qos
        )

        now_ns = self.get_clock().now().nanoseconds
        self.last_control_ns = now_ns
        self.last_cmd_ns = 0
        self.last_torque_ns = 0
        self.last_odom_publish_ns = 0
        self.last_torque_status_publish_ns = 0
        self.requested_linear = 0.0
        self.requested_angular = 0.0
        self.override_torque = [0.0, 0.0]
        self.applied_effort = [0.0, 0.0]
        self.wheel_velocity = [0.0, 0.0]
        self.have_wheel_state = False
        self.target_velocity = [0.0, 0.0]
        self.error_integral = [0.0, 0.0]
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        self.timer = self.create_timer(1.0 / self.control_rate, self._control_update)
        self.get_logger().info(
            "Effort drive ready: /cmd_vel and /wheel_torque_commands -> "
            "/wheel_effort_controller/commands"
        )

    def _cmd_vel_callback(self, message: Twist) -> None:
        if not isfinite(message.linear.x) or not isfinite(message.angular.z):
            self.get_logger().error("Ignoring non-finite /cmd_vel command")
            return
        self.requested_linear = float(message.linear.x)
        self.requested_angular = float(message.angular.z)
        self.last_cmd_ns = self.get_clock().now().nanoseconds

    def _torque_callback(self, message: Float64MultiArray) -> None:
        if len(message.data) != 2:
            self.get_logger().error(
                "/wheel_torque_commands requires [left_Nm, right_Nm]"
            )
            return
        if not all(isfinite(value) for value in message.data):
            self.get_logger().error("Ignoring non-finite wheel torque command")
            return
        self.override_torque = [
            clamp(float(message.data[0]), -self.max_torque, self.max_torque),
            clamp(float(message.data[1]), -self.max_torque, self.max_torque),
        ]
        self.last_torque_ns = self.get_clock().now().nanoseconds
        self.last_cmd_ns = 0
        self.requested_linear = 0.0
        self.requested_angular = 0.0
        self.target_velocity = [0.0, 0.0]
        self.error_integral = [0.0, 0.0]

    def _joint_state_callback(self, message: JointState) -> None:
        velocity_by_name = dict(zip(message.name, message.velocity))
        if self.left_joint not in velocity_by_name or self.right_joint not in velocity_by_name:
            return
        self.wheel_velocity[0] = float(velocity_by_name[self.left_joint])
        self.wheel_velocity[1] = float(velocity_by_name[self.right_joint])
        self.have_wheel_state = True

    def _control_update(self) -> None:
        now = self.get_clock().now()
        now_ns = now.nanoseconds
        dt = (now_ns - self.last_control_ns) * 1.0e-9
        self.last_control_ns = now_ns
        if dt <= 0.0 or dt > 0.25:
            dt = 1.0 / self.control_rate

        torque_override_active = (
            self.last_torque_ns > 0
            and (now_ns - self.last_torque_ns) * 1.0e-9 <= self.torque_timeout
        )
        if torque_override_active:
            efforts = list(self.override_torque)
        elif not self.have_wheel_state:
            efforts = [0.0, 0.0]
        else:
            command_is_fresh = (
                self.last_cmd_ns > 0
                and (now_ns - self.last_cmd_ns) * 1.0e-9 <= self.command_timeout
            )
            linear = self.requested_linear if command_is_fresh else 0.0
            angular = self.requested_angular if command_is_fresh else 0.0
            if abs(linear) < 1.0e-9 and abs(angular) < 1.0e-9:
                # Do not let stored integral torque push the robot after stop.
                self.error_integral = [0.0, 0.0]
            requested_targets = wheel_angular_targets(
                linear, angular, self.wheel_radius, self.wheel_separation
            )
            max_step = self.max_wheel_acceleration * dt
            efforts = []
            for index, requested in enumerate(requested_targets):
                requested = clamp(requested, -self.max_wheel_speed, self.max_wheel_speed)
                delta = clamp(
                    requested - self.target_velocity[index], -max_step, max_step
                )
                self.target_velocity[index] += delta
                error = self.target_velocity[index] - self.wheel_velocity[index]
                self.error_integral[index] = clamp(
                    self.error_integral[index] + error * dt,
                    -self.integral_limit,
                    self.integral_limit,
                )
                effort = self.kp * error + self.ki * self.error_integral[index]
                efforts.append(
                    clamp(effort, -self.max_velocity_torque, self.max_velocity_torque)
                )

        efforts = limit_effort_commands(
            efforts,
            self.applied_effort,
            self.wheel_velocity,
            dt,
            self.max_torque,
            self.max_effort_rate,
            self.max_wheel_speed,
        )
        self.applied_effort = efforts

        command = Float64MultiArray()
        command.data = efforts
        self.effort_publisher.publish(command)
        torque_status_period_ns = int(1.0e9 / self.torque_status_publish_rate)
        if now_ns - self.last_torque_status_publish_ns >= torque_status_period_ns:
            self.applied_torque_publisher.publish(command)
            self.last_torque_status_publish_ns = now_ns

        if self.have_wheel_state:
            self.x, self.y, self.yaw, linear, angular = integrate_wheel_odometry(
                self.x,
                self.y,
                self.yaw,
                self.wheel_velocity[0],
                self.wheel_velocity[1],
                self.wheel_radius,
                self.wheel_separation,
                dt,
            )
            odom_period_ns = int(1.0e9 / self.odom_publish_rate)
            if now_ns - self.last_odom_publish_ns >= odom_period_ns:
                self._publish_odometry(now, linear, angular)
                self.last_odom_publish_ns = now_ns

    def _publish_odometry(self, stamp, linear: float, angular: float) -> None:
        half_yaw = 0.5 * self.yaw
        orientation_z = sin(half_yaw)
        orientation_w = cos(half_yaw)

        message = Odometry()
        message.header.stamp = stamp.to_msg()
        message.header.frame_id = self.odom_frame
        message.child_frame_id = self.base_frame
        message.pose.pose.position.x = self.x
        message.pose.pose.position.y = self.y
        message.pose.pose.orientation.z = orientation_z
        message.pose.pose.orientation.w = orientation_w
        message.twist.twist.linear.x = linear
        message.twist.twist.angular.z = angular
        message.pose.covariance[0] = 0.02
        message.pose.covariance[7] = 0.02
        message.pose.covariance[14] = 1.0e6
        message.pose.covariance[21] = 1.0e6
        message.pose.covariance[28] = 1.0e6
        message.pose.covariance[35] = 0.05
        message.twist.covariance[0] = 0.02
        message.twist.covariance[7] = 0.02
        message.twist.covariance[14] = 1.0e6
        message.twist.covariance[21] = 1.0e6
        message.twist.covariance[28] = 1.0e6
        message.twist.covariance[35] = 0.05
        self.odom_publisher.publish(message)

        if self.tf_broadcaster is not None:
            transform = TransformStamped()
            transform.header = message.header
            transform.child_frame_id = self.base_frame
            transform.transform.translation.x = self.x
            transform.transform.translation.y = self.y
            transform.transform.rotation.z = orientation_z
            transform.transform.rotation.w = orientation_w
            self.tf_broadcaster.sendTransform(transform)

    def stop(self) -> None:
        if not self.context.ok():
            return
        message = Float64MultiArray()
        message.data = [0.0, 0.0]
        try:
            self.effort_publisher.publish(message)
        except RuntimeError:
            # SIGINT may invalidate the context between ok() and publish().
            pass


def main(args=None) -> None:
    rclpy.init(args=args)
    node = EffortDrive()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

"""Print a compact live view of Nino's IMU, wheel encoders, and lidar."""

from math import isfinite

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, JointState, LaserScan


class SensorMonitor(Node):
    """Report sensor values and measured receive rates once per second."""

    def __init__(self) -> None:
        super().__init__("sensor_monitor")
        self.declare_parameter("left_wheel_joint", "left_wheel_joint")
        self.declare_parameter("right_wheel_joint", "right_wheel_joint")

        self.left_joint = str(self.get_parameter("left_wheel_joint").value)
        self.right_joint = str(self.get_parameter("right_wheel_joint").value)
        self.imu: Imu | None = None
        self.encoders: JointState | None = None
        self.scan: LaserScan | None = None
        self.counts = {"imu": 0, "encoders": 0, "scan": 0}
        self.report_number = 0

        self.create_subscription(
            Imu, "/imu/data", self._imu_callback, qos_profile_sensor_data
        )
        self.create_subscription(
            JointState,
            "/joint_states",
            self._encoder_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            LaserScan, "/scan", self._scan_callback, qos_profile_sensor_data
        )
        self.create_timer(1.0, self._report)
        self.get_logger().info(
            "Watching IMU /imu/data, encoders /joint_states, and lidar /scan"
        )

    def _imu_callback(self, message: Imu) -> None:
        self.imu = message
        self.counts["imu"] += 1

    def _encoder_callback(self, message: JointState) -> None:
        self.encoders = message
        self.counts["encoders"] += 1

    def _scan_callback(self, message: LaserScan) -> None:
        self.scan = message
        self.counts["scan"] += 1

    @staticmethod
    def _joint_value(message: JointState, joint: str) -> tuple[float, float] | None:
        try:
            index = message.name.index(joint)
        except ValueError:
            return None
        if index >= len(message.position) or index >= len(message.velocity):
            return None
        return float(message.position[index]), float(message.velocity[index])

    def _report(self) -> None:
        self.report_number += 1
        lines = []

        if self.imu is None:
            lines.append("IMU WAITING")
        else:
            gyro = self.imu.angular_velocity
            accel = self.imu.linear_acceleration
            lines.append(
                f"IMU {self.counts['imu']} Hz: "
                f"gyro=({gyro.x:+.3f},{gyro.y:+.3f},{gyro.z:+.3f}) rad/s "
                f"accel=({accel.x:+.3f},{accel.y:+.3f},{accel.z:+.3f}) m/s^2"
            )

        left = (
            self._joint_value(self.encoders, self.left_joint)
            if self.encoders is not None
            else None
        )
        right = (
            self._joint_value(self.encoders, self.right_joint)
            if self.encoders is not None
            else None
        )
        if left is None or right is None:
            lines.append("ENCODERS WAITING")
        else:
            lines.append(
                f"ENCODERS {self.counts['encoders']} Hz: "
                f"left pos={left[0]:+.3f} rad vel={left[1]:+.3f} rad/s; "
                f"right pos={right[0]:+.3f} rad vel={right[1]:+.3f} rad/s"
            )

        if self.scan is None:
            lines.append("LIDAR WAITING")
        else:
            valid = [
                value
                for value in self.scan.ranges
                if isfinite(value) and self.scan.range_min <= value <= self.scan.range_max
            ]
            nearest = min(valid) if valid else float("nan")
            lines.append(
                f"LIDAR {self.counts['scan']} Hz: "
                f"valid={len(valid)}/{len(self.scan.ranges)} nearest={nearest:.3f} m"
            )

        message = " | ".join(lines)
        if self.report_number >= 3 and "WAITING" in message:
            self.get_logger().warning(message)
        else:
            self.get_logger().info(message)
        self.counts = {"imu": 0, "encoders": 0, "scan": 0}


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SensorMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

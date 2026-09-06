"""ROS 2 transport shared by the Gym environment and deployed policy node."""

from __future__ import annotations

from copy import deepcopy
from threading import Lock
from time import monotonic, sleep

import rclpy
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from ros_gz_interfaces.srv import ControlWorld
from sensor_msgs.msg import Imu, JointState, LaserScan
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger

from nino_rl.core import RobotState, quaternion_to_euler


SENSOR_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
)


class RosRobotInterface(Node):
    def __init__(
        self,
        world_name: str = "long_hall",
        subscribe_plan: bool = False,
        use_sim_time: bool = True,
        node_name: str = "nino_rl_interface",
        plan_topic: str = "/plan",
    ) -> None:
        super().__init__(
            node_name,
            parameter_overrides=[
                rclpy.parameter.Parameter("use_sim_time", value=use_sim_time)
            ],
        )
        self._lock = Lock()
        self._state = RobotState()
        self._received = set()
        self._nav_path: tuple[str, list[tuple[float, float]]] | None = None

        self.torque_publisher = self.create_publisher(
            Float64MultiArray, "/wheel_torque_commands", 10
        )
        self.create_subscription(Odometry, "/odom", self._odom_callback, 10)
        self.create_subscription(Imu, "/imu/data", self._imu_callback, SENSOR_QOS)
        self.create_subscription(JointState, "/joint_states", self._joint_callback, SENSOR_QOS)
        self.create_subscription(LaserScan, "/scan", self._scan_callback, SENSOR_QOS)
        if subscribe_plan:
            self.create_subscription(Path, plan_topic, self._plan_callback, 10)

        self.world_control = self.create_client(
            ControlWorld, f"/world/{world_name}/control"
        )
        self.reset_odometry = self.create_client(Trigger, "/reset_wheel_odometry")

    def _odom_callback(self, message: Odometry) -> None:
        quaternion = message.pose.pose.orientation
        _, _, yaw = quaternion_to_euler(
            quaternion.x, quaternion.y, quaternion.z, quaternion.w
        )
        with self._lock:
            self._state.x = float(message.pose.pose.position.x)
            self._state.y = float(message.pose.pose.position.y)
            self._state.yaw = yaw
            self._state.linear_velocity = float(message.twist.twist.linear.x)
            self._state.yaw_rate = float(message.twist.twist.angular.z)
            self._received.add("odom")

    def _imu_callback(self, message: Imu) -> None:
        quaternion = message.orientation
        roll, pitch, _ = quaternion_to_euler(
            quaternion.x, quaternion.y, quaternion.z, quaternion.w
        )
        with self._lock:
            self._state.roll = roll
            self._state.pitch = pitch
            self._state.gyro_z = float(message.angular_velocity.z)
            self._state.accel_x = float(message.linear_acceleration.x)
            self._state.accel_z = float(message.linear_acceleration.z)
            self._received.add("imu")

    def _joint_callback(self, message: JointState) -> None:
        velocity = dict(zip(message.name, message.velocity))
        if "left_wheel_joint" not in velocity or "right_wheel_joint" not in velocity:
            return
        with self._lock:
            self._state.left_wheel_velocity = float(velocity["left_wheel_joint"])
            self._state.right_wheel_velocity = float(velocity["right_wheel_joint"])
            self._received.add("joint")

    def _scan_callback(self, message: LaserScan) -> None:
        with self._lock:
            self._state.lidar_ranges = tuple(float(value) for value in message.ranges)
            self._state.lidar_range_max = float(message.range_max)
            self._received.add("scan")

    def _plan_callback(self, message: Path) -> None:
        points = [(pose.pose.position.x, pose.pose.position.y) for pose in message.poses]
        if len(points) >= 2:
            with self._lock:
                self._nav_path = (message.header.frame_id or "odom", points)

    def snapshot(self) -> RobotState:
        with self._lock:
            return deepcopy(self._state)

    def nav_path(self) -> tuple[str, list[tuple[float, float]]] | None:
        with self._lock:
            return deepcopy(self._nav_path)

    def sensors_ready(self) -> bool:
        with self._lock:
            return self._received == {"odom", "imu", "joint", "scan"}

    def wait_for_sensors(self, timeout: float) -> None:
        deadline = monotonic() + timeout
        while monotonic() < deadline:
            if self.sensors_ready():
                return
            sleep(0.05)
        with self._lock:
            missing = sorted({"odom", "imu", "joint", "scan"} - self._received)
        raise TimeoutError(
            "Không nhận đủ dữ liệu ROS trong thời gian chờ; thiếu: " + ", ".join(missing)
        )

    def publish_torque(self, left_nm: float, right_nm: float) -> None:
        message = Float64MultiArray()
        message.data = [float(left_nm), float(right_nm)]
        self.torque_publisher.publish(message)

    @staticmethod
    def _wait_future(future, timeout: float):
        deadline = monotonic() + timeout
        while monotonic() < deadline:
            if future.done():
                exception = future.exception()
                if exception is not None:
                    raise RuntimeError(str(exception)) from exception
                return future.result()
            sleep(0.01)
        raise TimeoutError("ROS service call timed out")

    def reset_episode(self, timeout: float = 8.0) -> None:
        """Reset Gazebo models, then reset wheel odometry/controller memory."""
        for _ in range(5):
            self.publish_torque(0.0, 0.0)
            sleep(0.02)
        if not self.world_control.wait_for_service(timeout_sec=timeout):
            raise TimeoutError(
                f"Missing {self.world_control.srv_name}; start training_sim.launch.py"
            )
        request = ControlWorld.Request()
        request.world_control.reset.model_only = True
        response = self._wait_future(self.world_control.call_async(request), timeout)
        if response is None or not response.success:
            raise RuntimeError("Gazebo rejected the model-only episode reset")

        if not self.reset_odometry.wait_for_service(timeout_sec=timeout):
            raise TimeoutError("Missing /reset_wheel_odometry from effort_drive")
        response = self._wait_future(
            self.reset_odometry.call_async(Trigger.Request()), timeout
        )
        if response is None or not response.success:
            message = response.message if response is not None else "no response"
            raise RuntimeError(f"Wheel odometry reset failed: {message}")
        sleep(0.30)

"""Mandatory live-system checks run before any PPO training starts."""

from __future__ import annotations

import argparse
from math import isfinite
from pathlib import Path
from threading import Event, Thread
from time import monotonic, sleep

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from nino_rl.core import load_config
from nino_rl.ros_interface import RosRobotInterface


class PreflightNode(RosRobotInterface):
    def __init__(self, plan_topic: str, nav_cmd_topic: str) -> None:
        super().__init__(
            world_name="long_hall",
            subscribe_plan=True,
            node_name="nino_rl_preflight",
            plan_topic=plan_topic,
            nav_cmd_topic=nav_cmd_topic,
        )
        latched = QoSProfile(depth=1)
        latched.reliability = ReliabilityPolicy.RELIABLE
        latched.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.map_message = None
        self.amcl_pose = None
        self.create_subscription(OccupancyGrid, "/map", self._map_callback, latched)

    def _map_callback(self, message: OccupancyGrid) -> None:
        self.map_message = message

    def _amcl_callback(self, message: PoseWithCovarianceStamped) -> None:
        self.amcl_pose = message
        super()._amcl_callback(message)


def _wait_until(predicate, timeout: float, failure: str) -> None:
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        if predicate():
            return
        sleep(0.05)
    raise RuntimeError(failure)


def run_preflight(config: dict, timeout: float = 30.0) -> list[str]:
    """Run all twelve required checks or raise before creating a trainer."""
    owns_rclpy = not rclpy.ok()
    if owns_rclpy:
        rclpy.init(args=[])
    nav = config["navigation"]
    node = PreflightNode(
        str(nav.get("plan_topic", "/plan")),
        str(nav.get("nav_cmd_topic", "/cmd_vel_nav")),
    )
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    stop = Event()

    def spin() -> None:
        while not stop.is_set() and rclpy.ok():
            executor.spin_once(timeout_sec=0.05)

    thread = Thread(target=spin, daemon=True)
    thread.start()
    passed: list[str] = []
    try:
        node.wait_for_sensors(timeout)
        # Preflight is a controlled flat-ground probe, including when a prior
        # training run left terrain or the robot at a different pose.
        node.configure_training_cables([], timeout=timeout)
        node.reset_episode(
            timeout=timeout,
            start_pose=tuple(float(v) for v in nav["start_pose"]),
        )
        state = node.snapshot()
        passed.append("1 robot spawned and sensor graph is connected")

        if not all(
            isfinite(value)
            for value in (
                state.orientation_x,
                state.orientation_y,
                state.orientation_z,
                state.orientation_w,
                state.gyro_x,
                state.gyro_y,
                state.gyro_z,
                state.accel_x,
                state.accel_y,
                state.accel_z,
            )
        ):
            raise RuntimeError("IMU contains non-finite values")
        passed.append("3 IMU publishes finite orientation, gyro, and acceleration")

        if not all(
            isfinite(v)
            for v in (state.left_wheel_velocity, state.right_wheel_velocity)
        ):
            raise RuntimeError("joint_states contains invalid wheel velocities")
        passed.append("4 left/right joint states are valid")
        passed.append("5 odometry is valid")

        finite_scan = [value for value in state.lidar_ranges if isfinite(value)]
        if not finite_scan or min(finite_scan) < 0.0:
            raise RuntimeError("laser scan has no finite non-negative range")
        passed.append("6 laser scan is valid")

        _wait_until(
            lambda: node.map_message is not None
            and node.map_message.info.width > 0
            and node.map_message.info.height > 0,
            timeout,
            "map_server did not publish a non-empty /map",
        )
        passed.append("8 existing/static map loaded")

        start = tuple(float(v) for v in nav["start_pose"])
        goal = tuple(float(v) for v in nav["goal_pose"])
        node.set_initial_pose(*start, timeout=timeout)
        passed.append("9 AMCL localization is active")
        node.send_navigation_goal(*goal, timeout=timeout)
        node.wait_for_navigation(timeout, float(nav["stale_seconds"]))
        passed.append("10 Nav2 generated a hallway path")
        passed.append("11 Nav2 local velocity reference and path are observable by RL")

        required_tf = [
            ("map", "odom"),
            ("odom", "base_footprint"),
            ("base_footprint", "base_link"),
            ("base_link", "imu_link"),
            ("base_link", "laser"),
            ("base_link", "left_wheel_link"),
            ("base_link", "right_wheel_link"),
        ]
        for parent, child in required_tf:
            _wait_until(
                lambda p=parent, c=child: node.tf_buffer.can_transform(
                    p, c, rclpy.time.Time()
                ),
                timeout,
                f"missing TF {parent} -> {child}",
            )
        passed.append("7 TF tree map->odom->base_footprint->base_link->sensors/wheels is valid")

        before = node.snapshot()
        test_torque = min(1.0, 0.25 * float(config["max_wheel_torque_nm"]))
        max_left_change = 0.0
        max_right_change = 0.0
        actuation_deadline = monotonic() + min(timeout, 3.0)
        # Turn in place so the probe cannot push the robot down the hall. Keep
        # publishing until DDS discovery, the effort slew limiter, and static
        # wheel friction have all had time to settle.
        while monotonic() < actuation_deadline:
            node.publish_torque(test_torque, -test_torque)
            sleep(0.05)
            sample = node.snapshot()
            max_left_change = max(
                max_left_change,
                abs(sample.left_wheel_velocity - before.left_wheel_velocity),
            )
            max_right_change = max(
                max_right_change,
                abs(sample.right_wheel_velocity - before.right_wheel_velocity),
            )
            if max_left_change >= 0.05 and max_right_change >= 0.05:
                break
        node.publish_torque(0.0, 0.0)
        if max_left_change < 0.05 or max_right_change < 0.05:
            raise RuntimeError(
                "direct torque did not rotate both wheels "
                f"(max delta left={max_left_change:.3f}, "
                f"right={max_right_change:.3f} rad/s)"
            )
        passed.append("2 both wheel joints rotate under bounded direct torque")
        passed.append("12 RL torque interface physically controls both wheels")
        return sorted(passed, key=lambda value: int(value.split()[0]))
    finally:
        try:
            node.publish_torque(0.0, 0.0)
            node.cancel_navigation_goal()
        except RuntimeError:
            pass
        stop.set()
        thread.join(timeout=2.0)
        executor.remove_node(node)
        executor.shutdown(timeout_sec=2.0)
        node.destroy_node()
        if owns_rclpy and rclpy.ok():
            rclpy.shutdown()


def main() -> None:
    share = Path(get_package_share_directory("nino_rl"))
    parser = argparse.ArgumentParser(description="Verify Nav2-guided RL prerequisites")
    parser.add_argument("--config", type=Path, default=share / "config" / "ppo.yaml")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    try:
        results = run_preflight(load_config(args.config), args.timeout)
    except (RuntimeError, TimeoutError) as error:
        raise SystemExit(f"PREFLIGHT FAILED: {error}") from error
    for result in results:
        print(f"PASS: {result}")
    print("PASS: all 12 checks; training is allowed")


if __name__ == "__main__":
    main()

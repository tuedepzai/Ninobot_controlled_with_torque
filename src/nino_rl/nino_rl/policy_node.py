"""Deploy a trained PPO policy as a safe ROS 2 torque-control node."""

from __future__ import annotations

import argparse
from math import cos, sin
from pathlib import Path
import sys
from time import monotonic

from ament_index_python.packages import get_package_share_directory
import numpy as np
import rclpy
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener

from nino_rl.core import (
    OBSERVATION_SIZE,
    PathTracker,
    goal_reached,
    load_config,
    make_observation,
    quaternion_to_euler,
)
from nino_rl.ros_interface import RosRobotInterface


def arguments() -> argparse.Namespace:
    share = Path(get_package_share_directory("nino_rl"))
    parser = argparse.ArgumentParser(description="Run a trained Nino PPO torque policy")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=share / "config" / "ppo.yaml")
    parser.add_argument("--path", type=Path, default=share / "config" / "path.yaml")
    parser.add_argument("--plan-topic", default="/plan")
    parser.add_argument("--use-sim-time", action="store_true")
    parser.add_argument("--device", default=None, help="cuda, cpu, or auto")
    return parser.parse_args(sys.argv[1:])


class PolicyNode(RosRobotInterface):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(
            subscribe_plan=True,
            use_sim_time=args.use_sim_time,
            node_name="nino_rl_policy",
            plan_topic=args.plan_topic,
        )
        try:
            from stable_baselines3 import PPO
        except ImportError as error:
            raise RuntimeError("Thiếu stable-baselines3; xem README_VI.md") from error

        self.config = load_config(args.config)
        path_config = load_config(args.path)
        self.path = PathTracker(path_config["waypoints"])
        self.path_source = "YAML"
        self.last_nav_signature = None
        self.last_tf_warning = 0.0
        self.path_started_at = None
        self.deadline_reported = False
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.previous_action = np.zeros(2, dtype=np.float32)
        self.action_scale = float(self.config["max_wheel_torque_nm"])
        self.lookahead = list(self.config["path"]["lookahead_m"])
        self.collision_distance = float(self.config["lidar_collision_m"])
        self.off_path_limit = float(self.config["off_path_limit_m"])
        self.rollover_limit = np.deg2rad(float(self.config["rollover_limit_deg"]))
        device = args.device or str(self.config.get("device", "cuda"))
        self.model = PPO.load(args.model, device=device)
        expected = tuple(self.model.observation_space.shape)
        if expected != (OBSERVATION_SIZE,):
            raise ValueError(
                f"Policy observation shape {expected} không tương thích với "
                f"({OBSERVATION_SIZE},)"
            )
        self.timer = self.create_timer(1.0 / float(self.config["control_hz"]), self._control)
        self.get_logger().info(
            f"PPO policy loaded on {self.model.device}; waiting for sensors and {args.plan_topic}"
        )

    def _path_in_odom(self, frame: str, points: list[tuple[float, float]]):
        if frame == "odom":
            return points
        transform = self.tf_buffer.lookup_transform("odom", frame, Time())
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        _, _, yaw = quaternion_to_euler(rotation.x, rotation.y, rotation.z, rotation.w)
        c, s = cos(yaw), sin(yaw)
        return [
            (
                translation.x + c * x - s * y,
                translation.y + s * x + c * y,
            )
            for x, y in points
        ]

    def _update_nav_path(self) -> None:
        nav_path = self.nav_path()
        if nav_path is None:
            return
        frame, points = nav_path
        signature = (frame, len(points), hash(np.asarray(points, dtype=np.float64).tobytes()))
        if signature == self.last_nav_signature:
            return
        try:
            transformed = self._path_in_odom(frame, points)
        except TransformException as error:
            now = monotonic()
            if now - self.last_tf_warning >= 2.0:
                self.get_logger().warning(
                    f"Chưa đổi được /plan từ {frame} sang odom: {error}"
                )
                self.last_tf_warning = now
            return
        previous_goal = self.path.points[-1].copy()
        self.path.set_points(transformed)
        self.last_nav_signature = signature
        self.path_source = f"Nav2 ({frame}->odom)"
        goal_changed = np.linalg.norm(self.path.points[-1] - previous_goal) > float(
            self.config["goal_tolerance_m"]
        )
        if self.path_started_at is None or goal_changed:
            self.path_started_at = monotonic()
            self.deadline_reported = False
        self.get_logger().info(f"Using {len(points)} waypoints from {self.path_source}")

    def _control(self) -> None:
        if not self.sensors_ready():
            self.publish_torque(0.0, 0.0)
            return
        if self.path_started_at is None:
            self.path_started_at = monotonic()
        self._update_nav_path()
        state = self.snapshot()
        finite_ranges = [value for value in state.lidar_ranges if np.isfinite(value)]
        if finite_ranges and min(finite_ranges) <= self.collision_distance:
            self.publish_torque(0.0, 0.0)
            return
        observation, tracking = make_observation(
            state, self.path, self.lookahead, self.previous_action
        )
        timed_out = monotonic() - self.path_started_at >= float(
            self.config["max_episode_seconds"]
        )
        if (
            goal_reached(tracking, state, self.config)
            or timed_out
            or abs(tracking.lateral_error) >= self.off_path_limit
            or max(abs(state.roll), abs(state.pitch)) >= self.rollover_limit
        ):
            self.publish_torque(0.0, 0.0)
            self.previous_action.fill(0.0)
            if timed_out and not self.deadline_reported:
                self.get_logger().warning(
                    "Quá thời gian chạy tối đa; giữ mô-men hai bánh ở 0"
                )
                self.deadline_reported = True
            return
        action, _ = self.model.predict(observation, deterministic=True)
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        if not np.all(np.isfinite(action)):
            self.get_logger().error("Policy returned non-finite action; commanding zero torque")
            action = np.zeros(2, dtype=np.float32)
        self.publish_torque(
            float(action[0] * self.action_scale),
            float(action[1] * self.action_scale),
        )
        self.previous_action = action


def main() -> None:
    args = arguments()
    rclpy.init(args=[])
    node = None
    try:
        node = PolicyNode(args)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.publish_torque(0.0, 0.0)
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

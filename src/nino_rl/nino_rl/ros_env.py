"""Gymnasium environment backed by the live Nino Gazebo simulation."""

from __future__ import annotations

from copy import deepcopy
from math import ceil, degrees
from threading import Event, Thread
from time import sleep

import gymnasium as gym
import numpy as np
import rclpy
from gymnasium import spaces
from rclpy.executors import MultiThreadedExecutor

from nino_rl.core import (
    OBSERVATION_SIZE,
    PathTracker,
    RobotState,
    catmull_rom_path,
    compute_reward,
    goal_reached,
    is_wrong_direction,
    make_observation,
    metrics_dict,
)
from nino_rl.ros_interface import RosRobotInterface


class NinoGazeboEnv(gym.Env):
    """Continuous two-action environment: normalized left and right torque."""

    metadata = {"render_modes": []}

    def __init__(self, config: dict, total_training_steps: int = 1) -> None:
        super().__init__()
        self.config = config
        self.total_training_steps = max(1, int(total_training_steps))
        self.control_dt = 1.0 / float(config["control_hz"])
        self.max_steps = int(float(config["max_episode_seconds"]) / self.control_dt)
        self.action_scale = float(config["max_wheel_torque_nm"])
        self.lookahead = list(config["path"]["lookahead_m"])
        self.sensor_timeout = float(config["sensor_timeout_seconds"])
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -5.0, 5.0, shape=(OBSERVATION_SIZE,), dtype=np.float32
        )

        self._owns_rclpy = not rclpy.ok()
        if self._owns_rclpy:
            rclpy.init(args=[])
        self.ros = RosRobotInterface(world_name="long_hall")
        self.executor = MultiThreadedExecutor(num_threads=2)
        self.executor.add_node(self.ros)
        self.executor_stop = Event()
        self.executor_thread = Thread(target=self._spin_executor, daemon=True)
        self.executor_thread.start()
        self.ros.wait_for_sensors(self.sensor_timeout)

        self.global_steps = 0
        self.attempt_number = 0
        self.episode_steps = 0
        self.wrong_direction_steps = 0
        self.wrong_direction_required_steps = max(
            1,
            ceil(
                float(config["wrong_direction_hold_seconds"])
                / self.control_dt
            ),
        )
        self._randomization = {}
        self.path = self._make_curriculum_path()
        self.previous_action = np.zeros(2, dtype=np.float32)
        self.action_before_previous = np.zeros(2, dtype=np.float32)
        self.previous_tracking = None
        self.previous_robot_state = None
        self._last_noisy_state: RobotState | None = None
        self.episode_return = 0.0
        self.abs_lateral_sum = 0.0
        self.abs_roll_sum = 0.0
        self.abs_pitch_sum = 0.0
        self.imu_angular_xy_sum = 0.0
        self.imu_acceleration_change_sum = 0.0
        self.max_tilt_deg = 0.0

    def _spin_executor(self) -> None:
        while not self.executor_stop.is_set() and rclpy.ok():
            self.executor.spin_once(timeout_sec=0.05)

    def _curriculum_stage(self) -> tuple[int, float, float]:
        curriculum = self.config["curriculum"]
        goals = list(curriculum["goal_x_m"])
        if not curriculum.get("enabled", True):
            return len(goals) - 1, 1.0, float(goals[-1])
        fraction = min(1.0, self.global_steps / self.total_training_steps)
        stage = 0
        for index, boundary in enumerate(curriculum["stage_fractions"]):
            if fraction >= float(boundary):
                stage = index
        stage = min(stage, len(goals) - 1)
        level = stage / max(1, len(goals) - 1)
        return stage, level, float(goals[stage])

    def _make_curriculum_path(self) -> PathTracker:
        _, level, goal_x = self._curriculum_stage()
        spacing = float(self.config["path"]["point_spacing_m"])
        amplitude = float(self._randomization.get("path_amplitude", 0.0)) * level
        if amplitude > 1.0e-6:
            control_x = np.linspace(0.0, goal_x, 5)
            control_y = self.np_random.uniform(-amplitude, amplitude, size=5)
            control_y[0] = 0.0
            points = catmull_rom_path(zip(control_x, control_y), spacing)
            points[:, 1] = np.clip(points[:, 1], -0.75, 0.75)
            return PathTracker(points)
        x_values = np.arange(0.0, goal_x + 0.5 * spacing, spacing)
        if x_values[-1] < goal_x:
            x_values = np.append(x_values, goal_x)
        return PathTracker([(float(x), 0.0) for x in x_values])

    def _sample_randomization(self) -> None:
        cfg = self.config["domain_randomization"]
        if not cfg.get("enabled", True):
            self._randomization = {
                "traction": 1.0,
                "delay": 0.0,
                "torque_noise": 0.0,
                "position_noise": 0.0,
                "heading_noise": 0.0,
                "position_bias": 0.0,
                "heading_bias": 0.0,
                "dropout": 0.0,
                "path_amplitude": 0.0,
            }
            return

        uniform = lambda key: float(self.np_random.uniform(*cfg[key]))
        self._randomization = {
            "traction": uniform("traction_scale"),
            "delay": uniform("motor_delay_ms") / 1000.0,
            "torque_noise": uniform("torque_noise_std_nm"),
            "position_noise": uniform("position_noise_std_m"),
            "heading_noise": np.deg2rad(uniform("heading_noise_std_deg")),
            "position_bias": uniform("position_bias_m") * self.np_random.choice([-1.0, 1.0]),
            "heading_bias": np.deg2rad(uniform("heading_bias_deg")),
            "dropout": uniform("observation_dropout"),
            "path_amplitude": uniform("path_lateral_amplitude_m"),
        }

    def _noisy_state(self, truth: RobotState) -> RobotState:
        if (
            self._last_noisy_state is not None
            and self.np_random.random() < self._randomization["dropout"]
        ):
            return deepcopy(self._last_noisy_state)
        noisy = deepcopy(truth)
        sigma_position = self._randomization["position_noise"]
        noisy.x += self._randomization["position_bias"] + self.np_random.normal(0.0, sigma_position)
        noisy.y += self._randomization["position_bias"] + self.np_random.normal(0.0, sigma_position)
        noisy.yaw += self._randomization["heading_bias"] + self.np_random.normal(
            0.0, self._randomization["heading_noise"]
        )
        self._last_noisy_state = deepcopy(noisy)
        return noisy

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.ros.reset_episode()
        self.ros.wait_for_sensors(self.sensor_timeout)
        self.attempt_number += 1
        self.episode_steps = 0
        self.wrong_direction_steps = 0
        self.previous_action.fill(0.0)
        self.action_before_previous.fill(0.0)
        self._last_noisy_state = None
        self._sample_randomization()
        self.path = self._make_curriculum_path()
        self.episode_return = 0.0
        self.abs_lateral_sum = 0.0
        self.abs_roll_sum = 0.0
        self.abs_pitch_sum = 0.0
        self.imu_angular_xy_sum = 0.0
        self.imu_acceleration_change_sum = 0.0
        self.max_tilt_deg = 0.0
        truth = self.ros.snapshot()
        self.previous_robot_state = deepcopy(truth)
        observation, _ = make_observation(
            self._noisy_state(truth), self.path, self.lookahead, self.previous_action
        )
        _, self.previous_tracking = make_observation(
            truth, self.path, self.lookahead, self.previous_action
        )
        stage, level, goal_x = self._curriculum_stage()
        return observation, {
            "attempt": self.attempt_number,
            "curriculum_stage": stage + 1,
            "curriculum_level": level,
            "goal_x_m": goal_x,
        }

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        delay = min(self.control_dt * 0.8, self._randomization["delay"])
        if delay > 0.0:
            sleep(delay)
        torque = action.astype(np.float64) * self.action_scale * self._randomization["traction"]
        torque += self.np_random.normal(0.0, self._randomization["torque_noise"], size=2)
        torque = np.clip(torque, -self.action_scale, self.action_scale)
        self.ros.publish_torque(float(torque[0]), float(torque[1]))
        sleep(max(0.0, self.control_dt - delay))

        self.episode_steps += 1
        self.global_steps += 1
        elapsed = self.episode_steps * self.control_dt
        truth = self.ros.snapshot()
        observation, _ = make_observation(
            self._noisy_state(truth), self.path, self.lookahead, action
        )
        _, tracking = make_observation(truth, self.path, self.lookahead, action)

        min_lidar = min(truth.lidar_ranges, default=truth.lidar_range_max)
        succeeded = goal_reached(tracking, truth, self.config)
        rolled = max(abs(degrees(truth.roll)), abs(degrees(truth.pitch))) >= float(
            self.config["rollover_limit_deg"]
        )
        off_path = abs(tracking.lateral_error) >= float(self.config["off_path_limit_m"])
        collision = np.isfinite(min_lidar) and min_lidar <= float(self.config["lidar_collision_m"])
        wrong_direction_sample = (
            elapsed >= float(self.config["wrong_direction_grace_seconds"])
            and is_wrong_direction(tracking, truth, self.config)
        )
        if wrong_direction_sample:
            self.wrong_direction_steps += 1
        else:
            self.wrong_direction_steps = 0
        wrong_direction = (
            self.wrong_direction_steps >= self.wrong_direction_required_steps
        )
        timed_out = self.episode_steps >= self.max_steps
        terminated = bool(
            succeeded or rolled or wrong_direction or off_path or collision
        )
        truncated = bool(timed_out and not terminated)
        _, level, _ = self._curriculum_stage()
        reward, reward_terms = compute_reward(
            self.previous_tracking,
            tracking,
            truth,
            self.previous_robot_state,
            action,
            self.previous_action,
            self.action_before_previous,
            self.control_dt,
            elapsed,
            {
                **self.config["reward"],
                "target_finish_seconds": self.config["target_finish_seconds"],
                "goal_max_speed_m_s": self.config["goal_max_speed_m_s"],
                "goal_max_yaw_rate_rad_s": self.config[
                    "goal_max_yaw_rate_rad_s"
                ],
            },
            level,
            timed_out=truncated,
            succeeded=succeeded,
            wrong_direction=wrong_direction,
        )
        self.episode_return += reward
        self.abs_lateral_sum += abs(tracking.lateral_error)
        self.abs_roll_sum += abs(degrees(truth.roll))
        self.abs_pitch_sum += abs(degrees(truth.pitch))
        self.imu_angular_xy_sum += float(np.hypot(truth.gyro_x, truth.gyro_y))
        self.imu_acceleration_change_sum += float(
            np.linalg.norm(
                np.asarray(
                    [truth.accel_x, truth.accel_y, truth.accel_z], dtype=np.float64
                )
                - np.asarray(
                    [
                        self.previous_robot_state.accel_x,
                        self.previous_robot_state.accel_y,
                        self.previous_robot_state.accel_z,
                    ],
                    dtype=np.float64,
                )
            )
        )
        self.max_tilt_deg = max(
            self.max_tilt_deg, abs(degrees(truth.roll)), abs(degrees(truth.pitch))
        )
        self.previous_tracking = tracking
        self.previous_robot_state = deepcopy(truth)
        self.action_before_previous = self.previous_action.copy()
        self.previous_action = action.copy()

        info = {
            "attempt": self.attempt_number,
            "reward_terms": reward_terms,
            "applied_torque_nm": torque.tolist(),
            **metrics_dict(tracking, truth, elapsed, succeeded),
        }
        if terminated or truncated:
            count = max(1, self.episode_steps)
            info["episode_metrics"] = {
                **metrics_dict(tracking, truth, elapsed, succeeded),
                "attempt": self.attempt_number,
                "return": self.episode_return,
                "mean_abs_lateral_error_m": self.abs_lateral_sum / count,
                "mean_abs_roll_deg": self.abs_roll_sum / count,
                "mean_abs_pitch_deg": self.abs_pitch_sum / count,
                "mean_imu_angular_xy_rad_s": self.imu_angular_xy_sum / count,
                "mean_imu_acceleration_change_m_s2": (
                    self.imu_acceleration_change_sum / count
                ),
                "max_tilt_deg": self.max_tilt_deg,
                "finished_within_target_time": bool(
                    succeeded
                    and elapsed <= float(self.config["target_finish_seconds"])
                ),
                "time_margin_seconds": float(
                    self.config["target_finish_seconds"]
                )
                - elapsed,
                "target_finish_seconds": float(
                    self.config["target_finish_seconds"]
                ),
                "final_speed_m_s": truth.linear_velocity,
                "final_yaw_rate_rad_s": truth.yaw_rate,
                "wrong_direction_duration_seconds": (
                    self.wrong_direction_steps * self.control_dt
                ),
                "termination": (
                    "success" if succeeded else "rollover" if rolled else "wrong_direction" if wrong_direction else "off_path" if off_path else "collision" if collision else "timeout"
                ),
            }
            self.ros.publish_torque(0.0, 0.0)
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        try:
            self.ros.publish_torque(0.0, 0.0)
            sleep(0.05)
            self.executor_stop.set()
            self.executor_thread.join(timeout=2.0)
            self.executor.remove_node(self.ros)
            self.executor.shutdown(timeout_sec=2.0)
            self.ros.destroy_node()
        finally:
            if self._owns_rclpy and rclpy.ok():
                rclpy.shutdown()

from math import pi

import numpy as np

from nino_rl.core import (
    OBSERVATION_SIZE,
    PathTracker,
    RobotState,
    TrackingState,
    catmull_rom_path,
    compute_reward,
    lidar_sectors,
    make_observation,
    quaternion_to_euler,
    wrap_angle,
)


LOOKAHEAD = [0.5, 1.0, 2.0, 3.5, 5.0, 7.5, 10.0, 12.5, 15.0]


def test_path_projection_sign_and_lookahead():
    path = PathTracker([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)])
    path_s, lateral, heading, _ = path.project(3.0, 1.0)
    assert np.isclose(path_s, 3.0)
    assert np.isclose(lateral, 1.0)
    assert np.isclose(heading, 0.0)
    local, _, _, error = path.local_lookahead(3.0, 1.0, 0.0, [1.0])
    assert np.allclose(local, [[1.0, -1.0]])
    assert np.isclose(error, 0.0)


def test_quaternion_and_angle_wrapping():
    yaw = pi / 2.0
    roll, pitch, measured_yaw = quaternion_to_euler(0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2))
    assert np.allclose([roll, pitch, measured_yaw], [0.0, 0.0, yaw])
    assert np.isclose(wrap_angle(3.0 * pi), -pi)


def test_observation_has_fixed_finite_shape():
    state = RobotState(
        x=1.0,
        y=0.1,
        orientation_z=0.25,
        orientation_w=0.9682458,
        gyro_x=0.3,
        gyro_y=-0.2,
        gyro_z=0.1,
        accel_x=1.0,
        accel_y=-2.0,
        accel_z=9.8,
        lidar_ranges=[float("inf"), 2.0, float("nan"), 1.0, 3.0],
        lidar_range_max=10.0,
    )
    path = PathTracker([(0.0, 0.0), (30.0, 0.0)])
    observation, tracking = make_observation(state, path, LOOKAHEAD, [0.0, 0.0])
    assert observation.shape == (OBSERVATION_SIZE,)
    assert observation.dtype == np.float32
    assert np.all(np.isfinite(observation))
    assert np.isclose(tracking.lateral_error, 0.1)
    assert np.allclose(
        observation[24:34],
        [0.0, 0.0, 0.25, 0.9682458, 0.1, -0.2 / 3.0, 0.1 / 3.0, 0.1, -0.2, 0.98],
    )


def test_lidar_empty_and_non_finite_are_safe():
    assert np.allclose(lidar_sectors([], 10.0), np.ones(5))
    sectors = lidar_sectors([float("inf"), float("nan"), -1.0, 5.0, 2.0], 10.0)
    assert np.all((0.0 <= sectors) & (sectors <= 1.0))


def test_catmull_rom_path_preserves_endpoints_and_is_finite():
    controls = [(0.0, 0.0), (2.0, 0.4), (4.0, -0.3), (6.0, 0.0)]
    points = catmull_rom_path(controls, spacing=0.2)
    assert np.allclose(points[0], controls[0])
    assert np.allclose(points[-1], controls[-1])
    assert len(points) > len(controls)
    assert np.all(np.isfinite(points))


def test_reward_prefers_progress_and_penalizes_tracking_error():
    config = {
        "progress_weight": 20.0,
        "rough_progress_weight": 50.0,
        "alignment_weight": 16.0,
        "lateral_weight": 0.8,
        "heading_weight": 0.2,
        "direction_weight": 1.0,
        "smoothness_weight": 0.2,
        "sigma_lateral_m": 0.3,
        "sigma_heading_rad": 0.35,
        "stuck_progress_m": 0.01,
        "stuck_penalty": 10.0,
        "rollover_threshold_deg": 30.0,
        "rollover_weight": 20.0,
        "timeout_distance_weight": 10.0,
        "timeout_constant": 100.0,
        "success_bonus": 100.0,
        "time_penalty": 0.01,
        "imu_stability_weight": 10.0,
        "imu_vibration_weight": 0.2,
        "imu_tilt_sigma_rad": 0.2,
        "imu_angular_rate_sigma_rad_s": 1.0,
        "imu_acceleration_change_sigma_m_s2": 2.0,
    }
    previous = TrackingState(0.0, 0.0, 0.0, 10.0)
    good = TrackingState(0.1, 0.0, 0.0, 9.9)
    bad = TrackingState(0.1, 0.6, 0.7, 9.9)
    args = dict(
        state=RobotState(linear_velocity=1.0),
        previous_state=RobotState(linear_velocity=1.0),
        action=[0.2, 0.2],
        previous_action=[0.1, 0.1],
        action_before_previous=[0.0, 0.0],
        dt=0.1,
        elapsed=2.0,
        reward_config=config,
        curriculum_level=0.0,
        timed_out=False,
        succeeded=False,
    )
    good_reward, _ = compute_reward(previous, good, **args)
    bad_reward, _ = compute_reward(previous, bad, **args)
    assert good_reward > bad_reward
    assert good_reward > 0.0


def test_timeout_and_rollover_have_negative_terms():
    config = {
        "progress_weight": 20.0,
        "rough_progress_weight": 50.0,
        "alignment_weight": 16.0,
        "lateral_weight": 0.8,
        "heading_weight": 0.2,
        "direction_weight": 1.0,
        "smoothness_weight": 0.2,
        "sigma_lateral_m": 0.3,
        "sigma_heading_rad": 0.35,
        "stuck_progress_m": 0.01,
        "stuck_penalty": 10.0,
        "rollover_threshold_deg": 30.0,
        "rollover_weight": 20.0,
        "timeout_distance_weight": 10.0,
        "timeout_constant": 100.0,
        "success_bonus": 100.0,
        "time_penalty": 0.01,
        "imu_stability_weight": 10.0,
        "imu_vibration_weight": 0.2,
        "imu_tilt_sigma_rad": 0.2,
        "imu_angular_rate_sigma_rad_s": 1.0,
        "imu_acceleration_change_sigma_m_s2": 2.0,
    }
    tracking = TrackingState(0.0, 0.0, 0.0, 5.0)
    _, terms = compute_reward(
        tracking,
        tracking,
        RobotState(roll=np.deg2rad(40.0)),
        RobotState(),
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        0.1,
        2.0,
        config,
        1.0,
        timed_out=True,
        succeeded=False,
    )
    assert terms["rollover"] < 0.0
    assert terms["timeout"] == -150.0


def test_reward_prefers_stable_imu_and_desired_direction():
    config = {
        "progress_weight": 20.0,
        "rough_progress_weight": 50.0,
        "alignment_weight": 16.0,
        "lateral_weight": 0.8,
        "heading_weight": 0.2,
        "direction_weight": 1.0,
        "smoothness_weight": 0.2,
        "sigma_lateral_m": 0.3,
        "sigma_heading_rad": 0.35,
        "stuck_progress_m": 0.01,
        "stuck_penalty": 10.0,
        "rollover_threshold_deg": 30.0,
        "rollover_weight": 20.0,
        "timeout_distance_weight": 10.0,
        "timeout_constant": 100.0,
        "success_bonus": 100.0,
        "time_penalty": 0.01,
        "imu_stability_weight": 10.0,
        "imu_vibration_weight": 0.2,
        "imu_tilt_sigma_rad": 0.2,
        "imu_angular_rate_sigma_rad_s": 1.0,
        "imu_acceleration_change_sigma_m_s2": 2.0,
    }
    previous_tracking = TrackingState(0.0, 0.0, 0.0, 10.0)
    aligned = TrackingState(0.1, 0.0, 0.0, 9.9)
    misaligned = TrackingState(0.1, 0.0, 1.0, 9.9)
    previous_state = RobotState(accel_z=9.81)
    stable_state = RobotState(linear_velocity=1.0, accel_z=9.81)
    unstable_state = RobotState(
        linear_velocity=1.0,
        roll=0.25,
        pitch=-0.20,
        gyro_x=1.5,
        gyro_y=-1.0,
        accel_x=3.0,
        accel_z=12.0,
    )
    common = dict(
        action=[0.2, 0.2],
        previous_action=[0.2, 0.2],
        action_before_previous=[0.2, 0.2],
        dt=0.1,
        elapsed=2.0,
        reward_config=config,
        curriculum_level=1.0,
        timed_out=False,
        succeeded=False,
    )
    stable_reward, stable_terms = compute_reward(
        previous_tracking,
        aligned,
        stable_state,
        previous_state,
        **common,
    )
    unstable_reward, unstable_terms = compute_reward(
        previous_tracking,
        misaligned,
        unstable_state,
        previous_state,
        **common,
    )
    assert stable_reward > unstable_reward
    assert stable_terms["imu_stability"] > unstable_terms["imu_stability"]
    assert stable_terms["imu_vibration"] > unstable_terms["imu_vibration"]
    assert stable_terms["direction"] > unstable_terms["direction"]

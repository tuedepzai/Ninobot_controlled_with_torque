from math import isclose, pi

from nino_control.kinematics import (
    clamp,
    integrate_wheel_odometry,
    limit_effort_commands,
    wheel_angular_targets,
)


def test_clamp() -> None:
    assert clamp(2.0, -1.0, 1.0) == 1.0
    assert clamp(-2.0, -1.0, 1.0) == -1.0
    assert clamp(0.5, -1.0, 1.0) == 0.5


def test_straight_wheel_targets_are_equal() -> None:
    left, right = wheel_angular_targets(0.5, 0.0, 0.1, 0.4)
    assert left == right == 5.0


def test_rotation_wheel_targets_are_opposite() -> None:
    left, right = wheel_angular_targets(0.0, 1.0, 0.1, 0.4)
    assert left == -2.0
    assert right == 2.0


def test_straight_odometry() -> None:
    x, y, yaw, linear, angular = integrate_wheel_odometry(
        0.0, 0.0, 0.0, 5.0, 5.0, 0.1, 0.4, 2.0
    )
    assert isclose(x, 1.0)
    assert isclose(y, 0.0)
    assert isclose(yaw, 0.0)
    assert isclose(linear, 0.5)
    assert isclose(angular, 0.0)


def test_yaw_is_normalized() -> None:
    _, _, yaw, _, _ = integrate_wheel_odometry(
        0.0, 0.0, pi - 0.01, -1.0, 1.0, 0.1, 0.4, 1.0
    )
    assert -pi <= yaw <= pi


def test_effort_is_slew_limited() -> None:
    output = limit_effort_commands(
        [12.0, -12.0], [0.0, 0.0], [0.0, 0.0], 0.001, 12.0, 20.0, 14.0
    )
    assert output == [0.02, -0.02]


def test_soft_speed_limit_blocks_acceleration_but_allows_braking() -> None:
    output = limit_effort_commands(
        [0.8, -0.8], [0.8, -0.8], [14.0, 14.0], 0.001, 12.0, 20.0, 14.0
    )
    assert output == [0.0, -0.8]


def test_negative_soft_speed_limit_is_symmetric() -> None:
    output = limit_effort_commands(
        [-0.8, 0.8], [-0.8, 0.8], [-14.0, -14.0], 0.001, 12.0, 20.0, 14.0
    )
    assert output == [0.0, 0.8]

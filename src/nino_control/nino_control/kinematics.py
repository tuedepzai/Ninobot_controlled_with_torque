"""Small, ROS-independent differential-drive helpers."""

from math import atan2, cos, sin


def clamp(value: float, lower: float, upper: float) -> float:
    """Limit value to the inclusive range [lower, upper]."""
    return max(lower, min(upper, value))


def limit_effort_commands(
    requested: tuple[float, float] | list[float],
    previous: tuple[float, float] | list[float],
    wheel_velocity: tuple[float, float] | list[float],
    dt: float,
    max_effort: float,
    max_effort_rate: float,
    soft_speed_limit: float,
) -> list[float]:
    """Slew-limit wheel effort and prevent torque into a speed limit.

    Torque opposite to wheel motion remains available for braking.  The soft
    limit must be lower than the joint's hard URDF limit so controller_manager
    never needs to reject an otherwise valid effort command.
    """
    max_step = max_effort_rate * max(dt, 0.0)
    output = []
    for target, old, velocity in zip(requested, previous, wheel_velocity):
        target = clamp(target, -max_effort, max_effort)
        effort = clamp(target, old - max_step, old + max_step)
        if velocity >= soft_speed_limit and effort > 0.0:
            effort = 0.0
        elif velocity <= -soft_speed_limit and effort < 0.0:
            effort = 0.0
        output.append(effort)
    return output


def wheel_angular_targets(
    linear_velocity: float,
    angular_velocity: float,
    wheel_radius: float,
    wheel_separation: float,
) -> tuple[float, float]:
    """Convert base velocity in m/s and rad/s to left/right wheel rad/s."""
    half_track = 0.5 * wheel_separation
    left = (linear_velocity - angular_velocity * half_track) / wheel_radius
    right = (linear_velocity + angular_velocity * half_track) / wheel_radius
    return left, right


def integrate_wheel_odometry(
    x: float,
    y: float,
    yaw: float,
    left_velocity: float,
    right_velocity: float,
    wheel_radius: float,
    wheel_separation: float,
    dt: float,
) -> tuple[float, float, float, float, float]:
    """Integrate wheel velocities and return pose, linear speed, yaw rate."""
    left_linear = left_velocity * wheel_radius
    right_linear = right_velocity * wheel_radius
    linear = 0.5 * (left_linear + right_linear)
    angular = (right_linear - left_linear) / wheel_separation
    delta_yaw = angular * dt
    midpoint_yaw = yaw + 0.5 * delta_yaw
    distance = linear * dt
    x += distance * cos(midpoint_yaw)
    y += distance * sin(midpoint_yaw)
    yaw = atan2(sin(yaw + delta_yaw), cos(yaw + delta_yaw))
    return x, y, yaw, linear, angular

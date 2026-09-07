"""Regression for AMCL corrections invalidating a frozen odometry endpoint."""

from math import cos, sin
from types import SimpleNamespace

import numpy as np

from nino_rl.core import PathTracker
from nino_rl.ros_interface import RosRobotInterface


def test_episode_path_uses_current_localization_without_accepting_new_plan():
    rotation = SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)
    translation = SimpleNamespace(x=0.0, y=0.0, z=0.0)
    transform = SimpleNamespace(
        transform=SimpleNamespace(translation=translation, rotation=rotation)
    )
    interface = SimpleNamespace(
        tf_buffer=SimpleNamespace(lookup_transform=lambda *_: transform),
        # A subsequent planner update is deliberately a different path.
        nav_path=lambda: ("map", [(20.0, 1.0), (30.0, 0.0)]),
    )
    episode_path = ("map", [(0.0, 0.0), (15.0, 0.0), (30.0, 0.0)])
    initial = RosRobotInterface.nav_path_in_odom(interface, episode_path)
    np.testing.assert_allclose(initial, episode_path[1])

    # AMCL shifts map->odom during the traverse. Both the stored path and the
    # robot's comparison coordinates must receive the same rigid transform.
    translation.x = -0.3
    translation.y = 0.5
    yaw = 0.04
    rotation.z, rotation.w = sin(yaw / 2), cos(yaw / 2)
    corrected = RosRobotInterface.nav_path_in_odom(interface, episode_path)
    expected = [
        (-0.3 + cos(yaw) * x - sin(yaw) * y,
         0.5 + sin(yaw) * x + cos(yaw) * y)
        for x, y in episode_path[1]
    ]
    np.testing.assert_allclose(corrected, expected)
    assert len(corrected) == 3
    assert np.isclose(PathTracker(corrected).total_length, 30.0)
    assert np.linalg.norm(np.asarray(corrected[-1]) - initial[-1]) > 1.0


def test_odometry_path_is_not_transformed_twice():
    interface = SimpleNamespace()
    path = ("odom", [(0.0, 0.0), (30.0, 0.0)])
    assert RosRobotInterface.nav_path_in_odom(interface, path) == path[1]


def test_cable_spawn_service_pose_matches_requested_floor_position():
    from math import pi
    import pytest
    from nino_rl.core import quaternion_to_euler
    from nino_rl.ros_interface import RosRobotInterface

    request = RosRobotInterface._cable_spawn_request("training_cable_0", 8.0, 0.008, 0.5)
    pose = request.entity_factory.pose
    assert pose.position.x == 8.0
    assert pose.position.y == 0.0
    assert pose.position.z == 0.008
    q = pose.orientation
    assert quaternion_to_euler(q.x, q.y, q.z, q.w) == pytest.approx((pi / 2, 0, 0.5))

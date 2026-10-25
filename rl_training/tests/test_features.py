from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features import calculate_reward, imu_features, path_features


def test_signed_error_progress_and_heading():
    e, heading, progress, remaining = path_features(0.5, -0.1, 0, [(0, 0), (1, 0)])
    assert (e, heading, progress, remaining) == pytest.approx((-0.1, 0, 0.5, 0.5))


def test_gravity_removed_at_rest_and_when_tilted():
    acceleration, roll, pitch = imu_features([0, 0, 9.80665], [0, 0, 0, 1])
    assert acceleration == pytest.approx([0, 0, 0])
    angle = 0.2
    q = [0, np.sin(angle/2), 0, np.cos(angle/2)]
    gravity = [-9.80665*np.sin(angle), 0, 9.80665*np.cos(angle)]
    acceleration, roll, pitch = imu_features(gravity, q)
    assert acceleration == pytest.approx([0, 0, 0], abs=1e-9)
    assert pitch == pytest.approx(angle)


def test_failure_dominates_stationary_reward():
    weights = dict(progress=20, cross_track=8, heading=0.5, vibration=0.02,
                   effort=0.01, smoothness=0.01, time=0.05, success=30, failure=-30)
    reward, terms = calculate_reward(0, 0, 0, 0, np.zeros(3), np.zeros(2),
                                     np.zeros(2), 0.1, None, weights)
    assert reward < 0  # Standing still must not collect positive reward.
    failed, _ = calculate_reward(0, 0.02, 0.21, 0, np.zeros(3), np.zeros(2),
                                np.zeros(2), 0.1, 'FAILED', weights)
    assert failed < -29

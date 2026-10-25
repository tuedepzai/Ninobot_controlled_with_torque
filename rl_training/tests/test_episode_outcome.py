from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from episode_outcome import InfrastructureFault, navigation_outcome


def test_odom_timeout_is_not_a_robot_failure_reward():
    with pytest.raises(InfrastructureFault, match='Odometry delivery timeout'):
        navigation_outcome(6, 65001, 'Odometry delivery timeout')


def test_deviation_is_still_task_failure():
    assert navigation_outcome(6, 1, 'Path deviation') == 'FAILED'


def test_success_requires_success_status_and_error_code():
    assert navigation_outcome(4, 0, '') == 'SUCCEEDED'
    assert navigation_outcome(4, 1, '') == 'FAILED'

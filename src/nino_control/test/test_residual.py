from math import nan

import pytest

from nino_control.residual import ResidualGate


def test_residual_is_disabled_by_default():
    gate = ResidualGate()
    gate.set_following(True)
    gate.update([0.3, -0.3], 1.0)
    assert gate.combine([0.2, 0.2], 1.1, True, 2.0)[0] == [0.2, 0.2]


def test_bounds_timeout_and_zero_velocity():
    gate = ResidualGate(enabled=True)
    gate.set_following(True)
    assert gate.update([100, -100], 1.0)
    assert gate.combine([0.1, 0.1], 1.1, True, 2.0)[0] == pytest.approx([0.4, -0.2])
    assert gate.combine([0.1, 0.1], 1.3, True, 2.0)[0] == [0.1, 0.1]
    assert gate.combine([0.1, 0.1], 1.1, False, 2.0)[0] == [0.1, 0.1]
    assert gate.combine([0.1, 0.1], 0.5, True, 2.0)[0] == [0.1, 0.1]


def test_invalid_message_and_episode_transition_clear_old_residual():
    gate = ResidualGate(enabled=True)
    gate.set_following(True)
    gate.update([0.2, 0.2], 1)
    assert not gate.update([nan, 0], 1.1)
    assert gate.combine([0, 0], 1.1, True, 2)[0] == [0, 0]
    gate.update([0.2, 0.2], 2)
    gate.set_following(False)
    gate.set_following(True)
    assert gate.combine([0, 0], 2.1, True, 2)[0] == [0, 0]


def test_total_operating_limit_preserved():
    gate = ResidualGate(enabled=True)
    gate.set_following(True)
    gate.update([0.3, -0.3], 1)
    total, residual = gate.combine([1.9, -1.9], 1.1, True, 2)
    assert total == [2, -2]
    assert residual == pytest.approx([0.1, -0.1])

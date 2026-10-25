"""ROS-independent bounded residual torque gate. Baseline control stays active."""

from math import isfinite

from nino_control.kinematics import clamp


class ResidualGate:
    def __init__(self, enabled=False, limit=0.3, timeout=0.2):
        if not all(isfinite(v) and v > 0 for v in (limit, timeout)):
            raise ValueError('Residual limit and timeout must be finite and positive')
        self.enabled, self.limit, self.timeout = enabled, limit, timeout
        self.following = False
        self.clear()

    def clear(self):
        self.command = (0.0, 0.0)
        self.received = None

    def set_following(self, following):
        if not following or following != self.following:
            self.clear()
        self.following = following

    def update(self, values, now):
        if len(values) != 2 or not all(isfinite(v) for v in values):
            self.clear()
            return False
        if not self.enabled or not self.following:
            self.clear()
            return False
        self.command = tuple(clamp(v, -self.limit, self.limit) for v in values)
        self.received = now
        return True

    def combine(self, baseline, now, moving, total_limit):
        active = (self.enabled and self.following and moving and self.received is not None
                  and 0 <= now-self.received <= self.timeout)
        residual = self.command if active else (0.0, 0.0)
        total = [clamp(b+r, -total_limit, total_limit) for b,r in zip(baseline, residual)]
        return total, [t-b for t,b in zip(total, baseline)]

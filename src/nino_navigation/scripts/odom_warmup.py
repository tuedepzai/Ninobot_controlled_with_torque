"""Require a continuous stream of advancing odometry before starting a task."""


class OdomWarmup:
    def __init__(self, duration=1.0, max_gap=0.25):
        self.duration, self.max_gap = duration, max_gap
        self.started = self.received = self.stamp = None
        self.samples = 0

    def observe(self, now, stamp):
        if self.stamp is not None and stamp == self.stamp:
            return False  # Repeated old messages do not prove freshness.
        if (self.received is None or now-self.received > self.max_gap
                or (self.stamp is not None and stamp < self.stamp)):
            self.started, self.samples = now, 0
        self.received, self.stamp = now, stamp
        self.samples += 1
        return True

    def ready(self, now):
        return (self.received is not None and now-self.received <= self.max_gap
                and self.samples >= 3 and self.received-self.started >= self.duration)

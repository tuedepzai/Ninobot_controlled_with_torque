"""Planar distance to an immutable polyline, independent of ROS."""

from math import hypot, isfinite


def cross_track_error(x, y, points):
    if not points or not all(isfinite(v) for p in points for v in p):
        raise ValueError('Reference path must contain finite points')
    if not isfinite(x) or not isfinite(y):
        raise ValueError('Robot position must be finite')
    best = hypot(x - points[0][0], y - points[0][1])
    for (ax, ay), (bx, by) in zip(points, points[1:]):
        dx, dy = bx - ax, by - ay
        length_sq = dx * dx + dy * dy
        t = max(0.0, min(1.0, ((x-ax)*dx + (y-ay)*dy) / length_sq)) if length_sq else 0.0
        best = min(best, hypot(x - (ax+t*dx), y - (ay+t*dy)))
    return best

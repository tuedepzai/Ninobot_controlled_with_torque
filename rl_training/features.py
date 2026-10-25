"""Features/reward in physical units, independent of ROS and the learner."""

from math import asin, atan2, hypot, isfinite, sin, cos

import numpy as np


def path_features(x, y, yaw, points):
    if len(points) < 2:
        raise ValueError('RL requires at least two reference points')
    nearest, prefix, best = None, 0.0, float('inf')
    for (ax, ay), (bx, by) in zip(points, points[1:]):
        dx, dy = bx-ax, by-ay
        length = hypot(dx, dy)
        if length < 1e-9:
            continue
        t = np.clip(((x-ax)*dx+(y-ay)*dy)/(length*length), 0, 1)
        px, py = ax+t*dx, ay+t*dy
        distance = hypot(x-px, y-py)
        if distance < best:
            best = distance
            heading = atan2(dy, dx)
            signed = ((x-px)*(-dy)+(y-py)*dx)/length
            nearest = (signed, atan2(sin(yaw-heading), cos(yaw-heading)), prefix+t*length)
        prefix += length
    if nearest is None:
        raise ValueError('Reference contains only zero-length segments')
    return (*nearest, prefix-nearest[2])


def imu_features(acceleration, quaternion):
    x, y, z, w = quaternion
    norm = (x*x+y*y+z*z+w*w)**0.5
    if not isfinite(norm) or norm < 1e-6:
        raise ValueError('IMU orientation unavailable')
    x, y, z, w = (v/norm for v in quaternion)
    roll = atan2(2*(w*x+y*z), 1-2*(x*x+y*y))
    pitch = asin(np.clip(2*(w*y-z*x), -1, 1))
    gravity = 9.80665*np.array([2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)])
    return np.asarray(acceleration)-gravity, roll, pitch


def calculate_reward(previous_progress, progress, cross_track, heading, acceleration,
                     action, previous_action, dt, outcome, weights):
    terms = {
        'progress': weights['progress']*float(np.clip(progress-previous_progress, -0.25, 0.25)),
        'cross_track': -weights['cross_track']*cross_track**2*dt,
        'heading': -weights['heading']*heading**2*dt,
        'vibration': -weights['vibration']*float(np.dot(acceleration, acceleration))*dt,
        'effort': -weights['effort']*float(np.dot(action, action))*dt,
        'smoothness': -weights['smoothness']*float(np.sum((action-previous_action)**2)),
        'time': -weights['time']*dt,
        'terminal': weights['success'] if outcome == 'SUCCEEDED' else
                    weights['failure'] if outcome else 0.0,
    }
    return sum(terms.values()), terms

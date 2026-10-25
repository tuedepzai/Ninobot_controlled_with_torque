#!/usr/bin/env python3
"""Run real SAC/Gazebo and force one PRE-GOAL lifecycle discovery failure.

Arguments are forwarded to train.py (use --mode smoke). No runtime failure or
accepted goal is retried. Normal use does not import this test hook.
"""
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ros_env import NinoResidualEnv
import train


original = NinoResidualEnv._wait_active
original_wait = NinoResidualEnv._wait
injected = False
fault_stage = 'lifecycle'


def with_one_startup_fault(self, names):
    global injected
    if fault_stage == 'lifecycle' and not injected and names[0] == 'controller_server':
        assert self.goal_handle is None
        injected = True
        previous = self.cfg['startup_progress_timeout']
        self.cfg['startup_progress_timeout'] = 0.4
        try:
            return original(self, ('injected_unavailable_controller',))
        finally:
            self.cfg['startup_progress_timeout'] = previous
    return original(self, names)


def with_one_initialpose_timeout(self, predicate, *args, **kwargs):
    global injected
    if (fault_stage == 'initialpose' and not injected
            and 'get_subscription_count' in getattr(getattr(predicate, '__code__', None), 'co_names', ())):
        assert self.goal_handle is None
        injected = True
        return original_wait(self, lambda: False, timeout=0.4)
    return original_wait(self, predicate, *args, **kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--fault-stage', choices=('lifecycle', 'initialpose'), default='lifecycle')
    options, remaining = parser.parse_known_args()
    fault_stage = options.fault_stage
    sys.argv = [sys.argv[0], *remaining]
    NinoResidualEnv._wait_active = with_one_startup_fault
    NinoResidualEnv._wait = with_one_initialpose_timeout
    train.main()
    assert injected, 'Startup recovery was not exercised'

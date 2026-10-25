from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ros_env import NinoResidualEnv, ObservationNotReady


def test_queued_sensor_callbacks_are_processed_before_observation():
    env = object.__new__(NinoResidualEnv)
    env.cfg = {'sensor_timeout': 0.75}
    attempts = []

    def observe():
        attempts.append(None)
        if len(attempts) < 3:
            raise ObservationNotReady('Waiting for queued TF/sensors')
        return 'fresh observation', {'fresh': True}

    def wait(predicate, timeout, residual):
        assert timeout == 0.75  # Do not weaken freshness limits for GPU training.
        assert residual == [0.1, -0.1]
        for _ in range(3):
            if predicate():
                return
        raise AssertionError('No fresh observation')

    env._observe, env._wait = observe, wait
    assert env._fresh_observation([0.1, -0.1]) == ('fresh observation', {'fresh': True})


def test_invalid_observation_is_not_silently_retried():
    env = object.__new__(NinoResidualEnv)
    env.cfg = {'sensor_timeout': 0.75}

    def invalid():
        raise RuntimeError('Invalid observation')

    env._observe = invalid
    env._wait = lambda predicate, **kwargs: predicate()
    with pytest.raises(RuntimeError, match='Invalid observation'):
        env._fresh_observation()


def test_ros_callbacks_keep_running_while_learner_is_busy():
    from threading import Thread
    from time import monotonic, sleep
    import rclpy
    from rclpy.context import Context
    from rclpy.clock import Clock, ClockType
    from rclpy.executors import SingleThreadedExecutor
    from ros_env import Bridge
    context = Context()
    rclpy.init(context=context)
    bridge = Bridge(context, 'rl_background_callback_test')
    executor = SingleThreadedExecutor(context=context)
    executor.add_node(bridge)
    bridge.create_timer(0.02, lambda: bridge.receive('test', object()),
                        clock=Clock(clock_type=ClockType.STEADY_TIME))
    thread = Thread(target=executor.spin, daemon=True)
    thread.start()
    try:
        sleep(1.0)  # Longer than the unchanged sensor timeout of 0.75 s.
        data, received, *_ = bridge.snapshot()
        assert 'test' in data
        assert monotonic()-received['test'] < 0.3
        bridge.receive('another', object())
        assert 'another' not in data  # Observation snapshots do not change underneath the learner.
    finally:
        executor.shutdown(timeout_sec=2)
        thread.join(timeout=2)
        assert not thread.is_alive()
        bridge.destroy_node()
        context.shutdown()


def test_callback_failure_is_propagated_instead_of_training_on_old_data(tmp_path):
    from episode_outcome import InfrastructureFault
    env = object.__new__(NinoResidualEnv)
    env.executor_error = ValueError('Reference changed')
    env.children = []
    env.cfg = {'startup_timeout': 1}
    env.session_dir = tmp_path
    with pytest.raises(InfrastructureFault, match='Reference changed'):
        env._wait(lambda: False)

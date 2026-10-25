from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from startup import StartupFault, retry_startup
from episode_outcome import InfrastructureFault


def test_retry_before_goal_preserves_operation_result():
    attempts, reports = [], []

    def operation():
        attempts.append(None)
        if len(attempts) == 1:
            raise StartupFault('/controller_server stuck inactive')
        return 'observation', {'planning_requests': 1}

    assert retry_startup(operation, 3, lambda *args: reports.append(args)) == (
        'observation', {'planning_requests': 1})
    assert len(attempts) == 2 and len(reports) == 1


def test_repeated_startup_failure_is_bounded_and_checkpointable():
    reports = []

    def operation():
        raise StartupFault('/planner_server inactive')

    with pytest.raises(InfrastructureFault, match='/planner_server'):
        retry_startup(operation, 3, lambda *args: reports.append(args))
    assert len(reports) == 3


def test_runtime_fault_does_not_silently_restart_episode():
    reports = []

    def operation():
        raise InfrastructureFault('Odometry lost after goal accepted')

    with pytest.raises(InfrastructureFault):
        retry_startup(operation, 3, lambda *args: reports.append(args))
    assert not reports


def test_configuration_error_is_not_retried():
    with pytest.raises(ValueError, match='bad config'):
        retry_startup(lambda: (_ for _ in ()).throw(ValueError('bad config')), 3,
                      lambda *args: pytest.fail('Unexpected retry'))


def test_inactive_node_timeout_reports_name_and_actual_state(monkeypatch):
    from types import SimpleNamespace
    from ros_env import NinoResidualEnv
    import ros_env
    clock = [0.0]

    def now():
        clock[0] += 0.1
        return clock[0]

    monkeypatch.setattr(ros_env, 'monotonic', now)
    env = object.__new__(NinoResidualEnv)
    env.cfg = {'startup_timeout': 10.0, 'startup_progress_timeout': 0.5}
    env.session_dir = Path('/test/episode')
    client = SimpleNamespace(service_is_ready=lambda: True, call_async=lambda request: None)
    env.bridge = SimpleNamespace(create_client=lambda *args: client, destroy_client=lambda c: None)
    env._result = lambda *args, **kwargs: SimpleNamespace(current_state=SimpleNamespace(id=2, label='inactive'))
    env._wait = lambda predicate: predicate()
    with pytest.raises(StartupFault, match=r'/controller_server.*inactive \(id=2\)'):
        env._wait_active(('controller_server',))


def test_process_exit_is_not_misreported_as_service_timeout():
    from types import SimpleNamespace
    from ros_env import NinoResidualEnv
    env = object.__new__(NinoResidualEnv)
    env.cfg = {'startup_timeout': 10.0, 'startup_progress_timeout': 5.0}
    env.session_dir = Path('/test/episode')
    client = SimpleNamespace(service_is_ready=lambda: True, call_async=lambda request: None)
    env.bridge = SimpleNamespace(create_client=lambda *args: client, destroy_client=lambda c: None)

    def result(*args, **kwargs):
        raise RuntimeError('Simulation/navigation process exited')

    env._result = result
    with pytest.raises(RuntimeError, match='process exited'):
        env._wait_active(('controller_server',))


def test_initialpose_timeout_is_retryable_after_localization_check(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import ros_env
    env = object.__new__(ros_env.NinoResidualEnv)
    env.closed = False
    env.episode_index = 0
    env.run_dir = tmp_path
    env.context = None
    env.executor = SimpleNamespace(add_node=lambda node: None)
    env.cfg = {'headless': True, 'max_path_deviation': 0.2, 'startup_progress_timeout': 15.0}
    env.residual_enabled = True
    env.map_path = Path('/test/map.yaml')
    calls = []
    env._cleanup = lambda: calls.append('cleanup')
    env._renew_ros_context = lambda: None
    env._start = lambda *args: None
    env._wait_active = lambda names: calls.append(names)
    monkeypatch.setattr(ros_env, 'Bridge', lambda *args: SimpleNamespace())

    def wait(predicate, **kwargs):
        if 'get_subscription_count' in predicate.__code__.co_names:
            assert ('map_server', 'amcl') in calls
            raise ros_env.RosWaitTimeout('missing initialpose subscriber')

    env._wait = wait
    with pytest.raises(StartupFault, match=r'AMCL /initialpose.*before goal submission'):
        env._reset_once()
    assert calls.count('cleanup') == 2


def test_clear_graph_timeout_is_bounded_retryable_infrastructure_fault(tmp_path):
    from types import SimpleNamespace
    from ros_env import NinoResidualEnv, RosWaitTimeout
    env = object.__new__(NinoResidualEnv)
    env.cfg = {'startup_progress_timeout': 15}
    env.session_dir = tmp_path
    env.bridge = SimpleNamespace(count_publishers=lambda topic: 1)
    env._wait = lambda *args, **kwargs: (_ for _ in ()).throw(RosWaitTimeout('stale graph'))
    reports = []
    with pytest.raises(InfrastructureFault, match='no simulation was started'):
        retry_startup(env._wait_clear_graph, 3, lambda *args: reports.append(args))
    assert len(reports) == 3


def test_clear_graph_does_not_ignore_external_publishers(tmp_path):
    from types import SimpleNamespace
    from ros_env import NinoResidualEnv
    env = object.__new__(NinoResidualEnv)
    env.cfg = {'startup_progress_timeout': 15}
    env.session_dir = tmp_path
    env.bridge = SimpleNamespace(count_publishers=lambda topic: 1 if topic == '/cmd_vel' else 0)
    def wait(predicate, timeout):
        assert predicate() is False
        assert timeout == 15
        from ros_env import RosWaitTimeout
        raise RosWaitTimeout('external cmd_vel publisher remains')
    env._wait = wait
    with pytest.raises(StartupFault) as fault:
        env._wait_clear_graph()
    assert "'/cmd_vel': 1" in str(fault.value)


def test_reset_uses_new_context_and_shuts_down_previous_participant():
    import rclpy
    from rclpy.context import Context
    from rclpy.executors import SingleThreadedExecutor
    from ros_env import NinoResidualEnv
    env = object.__new__(NinoResidualEnv)
    env.context = Context()
    rclpy.init(context=env.context)
    env.executor = SingleThreadedExecutor(context=env.context)
    previous = env.context
    try:
        env._renew_ros_context()
        assert not previous.ok()
        assert env.context is not previous
        assert env.context.ok()
    finally:
        env.executor.shutdown()
        env.context.shutdown()


def test_runtime_clock_timeout_is_infrastructure_fault_without_fake_reward():
    from types import SimpleNamespace
    import numpy as np
    from ros_env import NinoResidualEnv, RosWaitTimeout
    env = object.__new__(NinoResidualEnv)
    env.finished = False
    env.cfg = {'residual_limit': 0.3}
    env.residual_enabled = True
    env.last_action = np.zeros(2)
    env.session_dir = Path('/test/episode')
    env.bridge = SimpleNamespace(command=lambda values: None,
                                 get_clock=lambda: SimpleNamespace(
                                     now=lambda: SimpleNamespace(nanoseconds=0)))
    cleaned = []
    env._cleanup = lambda: cleaned.append(True)
    env._wait = lambda *args, **kwargs: (_ for _ in ()).throw(RosWaitTimeout('clock stopped'))
    with pytest.raises(InfrastructureFault, match='ROS data/clock timeout during episode'):
        env.step(np.zeros(2))
    assert cleaned == [True]

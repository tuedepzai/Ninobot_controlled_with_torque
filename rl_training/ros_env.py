"""Gymnasium environment owning one Gazebo + fixed-path Nav2 session.

Reset restarts only the child processes created here. No teleport-only reset:
encoder integration, PI state, TF and localization all start fresh together.
"""

from datetime import datetime
import hashlib
import json
from math import atan2
import os
from pathlib import Path
import signal
import subprocess
from threading import Lock, Thread
from time import monotonic, sleep

import gymnasium as gym
import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry, Path as PathMessage
from rcl_interfaces.srv import GetParameters
from rclpy.action import ActionClient
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float64MultiArray, String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener

from features import calculate_reward, imu_features, path_features
from episode_outcome import InfrastructureFault, navigation_outcome
from startup import StartupFault, retry_startup

OBSERVATIONS = [
    'cross_track', 'heading_error', 'remaining', 'roll', 'pitch',
    'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z',
    'speed', 'yaw_rate', 'cmd_speed', 'cmd_yaw_rate', 'wheel_left', 'wheel_right',
    'base_left', 'base_right', 'previous_action_left', 'previous_action_right',
    'residual_left', 'residual_right',
]


class ObservationNotReady(RuntimeError):
    """Data may become valid after the executor processes queued callbacks."""


class RosWaitTimeout(RuntimeError):
    pass


class Bridge(Node):
    def __init__(self, context, name):
        super().__init__(name, context=context,
                         parameter_overrides=[Parameter('use_sim_time', value=True)])
        self.data, self.received = {}, {}
        self.data_lock = Lock()
        self.state, self.reference, self.reference_hash = {}, None, None
        self.tf = Buffer()
        self.listener = TransformListener(self.tf, self)
        for topic, kind, key, qos in [
            ('/imu/data', Imu, 'imu', qos_profile_sensor_data),
            ('/joint_states', JointState, 'joints', qos_profile_sensor_data),
            ('/odom', Odometry, 'odom', 1), ('/cmd_vel', Twist, 'cmd', 1),
            ('/wheel_torque_base', Float64MultiArray, 'base', 1),
            ('/wheel_torque_applied', Float64MultiArray, 'total', 1),
            ('/wheel_residual_applied', Float64MultiArray, 'residual', 1),
        ]:
            self.create_subscription(kind, topic, lambda m, k=key: self.receive(k, m), qos)
        latch = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, '/episode/state', self.on_state, latch)
        self.create_subscription(PathMessage, '/episode/reference_path', self.on_path, latch)
        self.residual_pub = self.create_publisher(Float64MultiArray, '/wheel_torque_residual', 1)
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 1)
        self.navigator = ActionClient(self, NavigateToPose, '/navigate_to_pose')

    def receive(self, key, message):
        with self.data_lock:
            self.data[key], self.received[key] = message, monotonic()

    def snapshot(self):
        with self.data_lock:
            return self.data.copy(), self.received.copy(), self.reference, self.reference_hash, self.state.copy()

    def on_state(self, message):
        with self.data_lock:
            self.state = json.loads(message.data)

    def on_path(self, message):
        # Canonical geometry avoids transport-dependent CDR padding bytes.
        geometry = [message.header.frame_id, [
            [p.header.frame_id, p.pose.position.x, p.pose.position.y, p.pose.position.z,
             p.pose.orientation.x, p.pose.orientation.y, p.pose.orientation.z, p.pose.orientation.w]
            for p in message.poses]]
        digest = hashlib.sha256(json.dumps(geometry, separators=(',', ':')).encode()).hexdigest()
        if self.reference_hash is not None and digest != self.reference_hash:
            raise RuntimeError('Reference path changed within the RL episode')
        with self.data_lock:
            self.reference, self.reference_hash = message, digest

    def command(self, values):
        self.residual_pub.publish(Float64MultiArray(data=[float(v) for v in values]))


class NinoResidualEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, config, map_path, run_dir, residual_enabled=True):
        super().__init__()
        self.cfg = dict(config['environment'])
        self.cfg.setdefault('startup_progress_timeout', 15.0)
        self.cfg.setdefault('startup_attempts', 3)
        self.weights = dict(config['reward'])
        self.map_path = Path(map_path).expanduser().resolve()
        if not self.map_path.is_file():
            raise ValueError(f'Map not found: {self.map_path}')
        for name in ('residual_limit', 'step_seconds', 'max_episode_seconds', 'sensor_timeout',
                     'startup_timeout', 'startup_progress_timeout', 'max_path_deviation'):
            if not np.isfinite(self.cfg[name]) or self.cfg[name] <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if not isinstance(self.cfg['startup_attempts'], int) or self.cfg['startup_attempts'] < 1:
            raise ValueError('startup_attempts must be a positive integer')
        if self.cfg['residual_limit'] > 0.3:
            raise ValueError('First-stage residual_limit must be <= 0.3 N.m')
        self.run_dir = Path(run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.residual_enabled = residual_enabled
        self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-10.0, 10.0, (len(OBSERVATIONS),), dtype=np.float32)
        self.context = Context()
        rclpy.init(context=self.context)
        self.executor = SingleThreadedExecutor(context=self.context)
        self.executor_thread = None
        self.executor_error = None
        self.bridge = None
        self.children = []
        self.log_handles = []
        self.goal_handle = self.result_future = None
        self.closed = self.finished = False
        self.episode_index = 0
        self.records = None
        self.last_action = np.zeros(2, dtype=np.float32)

    def _start(self, name, command):
        stream = (self.session_dir/(name+'.log')).open('w')
        self.log_handles.append(stream)
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        self.children.append(child)

    def _wait(self, predicate, timeout=None, residual=None):
        deadline = monotonic()+(timeout or self.cfg['startup_timeout'])
        while not predicate():
            if monotonic() > deadline:
                raise RosWaitTimeout(f'ROS timeout; see {self.session_dir}')
            if any(c.poll() is not None for c in self.children):
                raise RuntimeError(f'Simulation/navigation process exited; see {self.session_dir}')
            if getattr(self, 'executor_error', None) is not None:
                raise InfrastructureFault(f'ROS callback failed: {self.executor_error}')
            if residual is not None:
                values = residual if self.bridge.state.get('state') == 'FOLLOWING' else (0.0, 0.0)
                self.bridge.command(values)
            if getattr(self, 'executor_thread', None) is not None:
                sleep(0.002)  # ROS callbacks continue while Torch computes.
            else:
                self.executor.spin_once(timeout_sec=0.02)

    def _spin_ros(self):
        try:
            self.executor.spin()
        except Exception as exc:
            self.executor_error = exc

    def _result(self, future, timeout=None):
        self._wait(future.done, timeout)
        return future.result()

    def _service_ready(self, service):
        self._wait(service.service_is_ready)

    def _renew_ros_context(self):
        # A new node on the old participant can inherit stale DDS graph
        # entries. Cleanup has already stopped/joined callbacks and destroyed
        # the old node. Replace the participant, not the learned policy.
        self.executor.shutdown()
        if self.context.ok():
            self.context.shutdown()
        self.context = Context()
        rclpy.init(context=self.context)
        self.executor = SingleThreadedExecutor(context=self.context)

    def _wait_clear_graph(self):
        try:
            self._wait(lambda: self.bridge.count_publishers('/clock') == 0
                       and self.bridge.count_publishers('/cmd_vel') == 0,
                       timeout=self.cfg['startup_progress_timeout'])
        except RosWaitTimeout as exc:
            counts = {topic: self.bridge.count_publishers(topic)
                      for topic in ('/clock', '/cmd_vel')}
            raise StartupFault(f'ROS graph still has publishers {counts}; '
                               'no simulation was started and no external nodes were stopped; '
                               f'see {self.session_dir}') from exc

    def _wait_active(self, names):
        from lifecycle_msgs.srv import GetState
        deadline = monotonic()+self.cfg['startup_timeout']
        stall_limit = self.cfg.get('startup_progress_timeout', 15.0)
        for name in names:
            client = self.bridge.create_client(GetState, '/'+name+'/get_state')
            last_state, last_change = None, monotonic()
            description = 'service unavailable'
            try:
                while True:
                    now = monotonic()
                    if now >= deadline or now-last_change >= stall_limit:
                        raise StartupFault(f'/{name} did not activate: {description}; '
                                           f'see {self.session_dir}/navigation.log')
                    if client.service_is_ready():
                        future = client.call_async(GetState.Request())
                        try:
                            reply = self._result(future, timeout=min(2.0, deadline-now))
                        except RosWaitTimeout:
                            future.cancel()
                            self.bridge.destroy_client(client)
                            client = self.bridge.create_client(GetState, '/'+name+'/get_state')
                            description = 'get_state response timed out'
                            continue
                        state = reply.current_state
                        description = f'{state.label} (id={state.id})'
                        if state.id == 3:
                            break
                        if state.id != last_state:
                            last_state, last_change = state.id, monotonic()
                    start = monotonic()
                    self._wait(lambda: monotonic()-start >= 0.2)
            finally:
                self.bridge.destroy_client(client)

    def _report_startup_retry(self, attempt, attempts, reason):
        event = dict(attempt=attempt, max_attempts=attempts, reason=reason,
                     session=str(self.session_dir), goal_submitted=False)
        with (self.run_dir/'startup_recovery.jsonl').open('a') as stream:
            stream.write(json.dumps(event)+'\n')
        print(f'Startup {attempt}/{attempts} failed before navigation: {reason}', flush=True)
        if attempt < attempts:
            print('Restarting owned simulation/Nav2; policy/replay are kept, no reward added.', flush=True)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return retry_startup(self._reset_once, self.cfg.get('startup_attempts', 3),
                             self._report_startup_retry)

    def _reset_once(self):
        if self.closed:
            raise RuntimeError('Environment is closed')
        self._cleanup()
        self._renew_ros_context()
        self.episode_index += 1
        self.session_dir = self.run_dir/f'episode_{self.episode_index:05d}_{datetime.now().strftime("%H%M%S_%f")}'
        self.session_dir.mkdir()
        self.bridge = Bridge(self.context, f'nino_rl_env_{self.episode_index}')
        self.executor.add_node(self.bridge)
        self.executor_error = None
        self.executor_thread = Thread(target=self._spin_ros, name='nino-rl-ros', daemon=True)
        self.executor_thread.start()
        goal_submitted = False
        startup_stage = 'ROS graph discovery'
        try:
            # Discover other sessions before creating any process of our own.
            start = monotonic()
            self._wait(lambda: monotonic()-start > 2.0)
            # DDS discovery can retain a just-stopped publisher briefly. Wait
            # for disposal announcements instead of calling it another live
            # simulation immediately after a reset. Never kill external nodes.
            self._wait_clear_graph()
            self._start('simulation', ['ros2', 'launch', 'nino_description', 'sim.launch.py',
                        'headless:='+str(self.cfg['headless']).lower(),
                        'enable_residual:='+str(self.residual_enabled).lower()])
            startup_stage = 'simulation odometry/IMU'
            self._wait(lambda: 'odom' in self.bridge.data and 'imu' in self.bridge.data)
            # Let control/bridge discovery settle before lifecycle bringup.
            settle = monotonic()
            self._wait(lambda: monotonic()-settle > 2.0)
            self._start('navigation', ['ros2', 'launch', 'nino_navigation', 'navigation.launch.py',
                        'rviz:=false', 'fixed_path:=true', 'map:='+str(self.map_path),
                        'max_path_deviation:='+str(self.cfg['max_path_deviation'])])
            # AMCL subscribes before activation and can discard an early pose.
            self._wait_active(('map_server', 'amcl'))
            startup_stage = 'AMCL /initialpose subscriber discovery'
            self._wait(lambda: self.bridge.pose_pub.get_subscription_count() > 0)
            initial = PoseWithCovarianceStamped()
            initial.header.frame_id = 'map'
            initial.pose.pose.orientation.w = 1.0
            initial.pose.covariance[0] = initial.pose.covariance[7] = 0.01
            initial.pose.covariance[35] = 0.01
            self.bridge.pose_pub.publish(initial)
            # Wait for all needed lifecycle nodes, rather than only discovery.
            self._wait_active(('controller_server', 'planner_server', 'velocity_smoother', 'collision_monitor'))
            startup_stage = 'effort_drive residual parameter service'
            params = self.bridge.create_client(GetParameters, '/effort_drive/get_parameters')
            self._service_ready(params)
            request = GetParameters.Request(names=['enable_residual', 'max_residual_torque'])
            values = self._result(params.call_async(request)).values
            if values[0].bool_value != self.residual_enabled or values[1].double_value < self.cfg['residual_limit']:
                raise RuntimeError('Residual control configuration mismatch')
            startup_stage = 'EpisodeManager action/state discovery'
            self._wait(lambda: self.bridge.navigator.server_is_ready()
                       and self.bridge.state.get('state') == 'IDLE')
            startup_stage = 'map to base_footprint transform'
            self._wait(lambda: self.bridge.tf.can_transform('map', 'base_footprint', Time()))
            # The manager has its own TF buffer; readiness in this node does not
            # imply that its buffer has received the same transforms yet.
            ready = self.bridge.create_client(Trigger, '/episode/ready')
            startup_stage = 'EpisodeManager readiness'
            self._service_ready(ready)
            deadline, consecutive = monotonic()+self.cfg['startup_timeout'], 0
            while consecutive < 3:
                reply = self._result(ready.call_async(Trigger.Request()))
                consecutive = consecutive+1 if reply.success else 0
                if monotonic() > deadline:
                    raise StartupFault(f'EpisodeManager not ready: {reply.message}; '
                                       f'see {self.session_dir}/navigation.log')
                start = monotonic()
                self._wait(lambda: monotonic()-start > 0.1)
            goal = NavigateToPose.Goal()
            goal.pose.header.frame_id = 'map'
            goal.pose.pose.position.x = self.cfg['goal_x']
            goal.pose.pose.position.y = self.cfg['goal_y']
            goal.pose.pose.orientation.w = 1.0
            # Once sent, even a lost acceptance response must not trigger a
            # startup retry: the server may already be executing this goal.
            goal_submitted = True
            self.goal_handle = self._result(self.bridge.navigator.send_goal_async(goal))
            if not self.goal_handle.accepted:
                raise RuntimeError('Episode rejected; inspect navigation.log')
            self.result_future = self.goal_handle.get_result_async()
            self._wait(lambda: self.bridge.state.get('state') == 'FOLLOWING' or self.result_future.done())
            if self.result_future.done():
                wrapped = self.result_future.result()
                navigation_outcome(wrapped.status, wrapped.result.error_code, wrapped.result.error_msg)
                raise RuntimeError(f'Navigation failed during reset: {wrapped}')
            self._wait(lambda: self.bridge.reference is not None and all(k in self.bridge.data for k in
                       ('imu', 'odom', 'joints', 'cmd', 'base', 'residual', 'total')))
            self.last_action = np.zeros(2, dtype=np.float32)
            observation, info = self._fresh_observation()
            self.start_sim = self.bridge.get_clock().now().nanoseconds*1e-9
            self.previous_progress = info['progress']
            self.finished = False
            self.records = (self.session_dir/'steps.jsonl').open('w')
            return observation, info
        except RosWaitTimeout as exc:
            self._cleanup()
            if not goal_submitted:
                raise StartupFault(f'{startup_stage} timed out before goal submission; '
                                   f'see {self.session_dir}') from exc
            raise InfrastructureFault(f'ROS timeout after goal submission; '
                                      f'see {self.session_dir}') from exc
        except BaseException:
            self._cleanup()
            raise

    def _observe(self):
        if self.executor_error is not None:
            raise InfrastructureFault(f'ROS callback failed: {self.executor_error}')
        b = self.bridge
        data, received, reference, reference_hash, state = b.snapshot()
        stale = [k for k in ('imu', 'joints', 'odom', 'cmd', 'base', 'residual', 'total')
                 if monotonic()-received.get(k, 0) > self.cfg['sensor_timeout']]
        if stale:
            raise ObservationNotReady('Sensor delivery timeout: '+', '.join(stale))
        try:
            transform = b.tf.lookup_transform('map', 'base_footprint', Time())
        except TransformException as exc:
            raise ObservationNotReady(str(exc)) from exc
        stamp = Time.from_msg(transform.header.stamp).nanoseconds
        age = (b.get_clock().now().nanoseconds-stamp)*1e-9
        if not -0.1 <= age <= self.cfg['sensor_timeout']:
            raise ObservationNotReady(f'Stale map pose: age={age:.3f}s')
        p, q = transform.transform.translation, transform.transform.rotation
        yaw = atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
        points = [(v.pose.position.x, v.pose.position.y) for v in reference.poses]
        error, heading, progress, remaining = path_features(p.x, p.y, yaw, points)
        imu = data['imu']
        a, orientation, w = imu.linear_acceleration, imu.orientation, imu.angular_velocity
        acceleration, roll, pitch = imu_features([a.x, a.y, a.z],
                                                [orientation.x, orientation.y, orientation.z, orientation.w])
        v, cmd = data['odom'].twist.twist, data['cmd']
        joint = dict(zip(data['joints'].name, data['joints'].velocity))
        left, right = joint['left_wheel_joint'], joint['right_wheel_joint']
        base = np.asarray(data['base'].data)
        accepted = np.asarray(data['residual'].data)
        observation = np.array([
            error/self.cfg['max_path_deviation'], heading/np.pi, remaining/10,
            roll/0.5, pitch/0.5, *(acceleration/10), w.x/3, w.y/3, w.z/3,
            v.linear.x/0.5, v.angular.z, cmd.linear.x/0.5, cmd.angular.z,
            left/12, right/12, *(base/2), *self.last_action,
            *(accepted/self.cfg['residual_limit']),
        ], dtype=np.float32)
        if observation.shape != self.observation_space.shape or not np.all(np.isfinite(observation)):
            raise RuntimeError('Invalid observation')
        info = dict(cross_track=error, heading_error=heading, progress=progress, remaining=remaining,
                    requested_goal=[self.cfg['goal_x'], self.cfg['goal_y']],
                    reference_goal=list(points[-1]),
                    acceleration=acceleration.tolist(), roll=roll, pitch=pitch,
                    base_torque=base.tolist(), residual_applied=accepted.tolist(),
                    total_torque=list(data['total'].data),
                    path_sha256=state.get('path_sha256'), reference_geometry_sha256=reference_hash,
                    planning_requests=state.get('planning_requests'))
        return np.clip(observation, -10, 10), info

    def _fresh_observation(self, residual=None):
        # CUDA/optimizer work can delay the single-threaded ROS executor. A
        # /clock callback alone can finish the step wait before sensor/TF
        # callbacks run. Pump callbacks until data passes the existing checks.
        captured, last_error = [], []

        def capture():
            try:
                captured.append(self._observe())
                return True
            except ObservationNotReady as exc:
                last_error[:] = [str(exc)]
                return False

        try:
            self._wait(capture, timeout=self.cfg['sensor_timeout'], residual=residual)
        except RosWaitTimeout as exc:
            raise RosWaitTimeout(f'{exc}; last observation error: {last_error}') from exc
        return captured[0]

    def step(self, action):
        if self.finished or self.bridge is None:
            raise RuntimeError('Call reset before step')
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (2,) or not np.all(np.isfinite(action)):
            self.bridge.command([0, 0])
            raise ValueError('Action must be a finite 2-vector')
        action = np.clip(action, -1, 1)
        physical = action*self.cfg['residual_limit'] if self.residual_enabled else np.zeros(2)
        before = self.bridge.get_clock().now().nanoseconds*1e-9
        previous_action = self.last_action.copy()
        self.bridge.command(physical)
        try:
            self._wait(lambda: self.result_future.done() or
                       self.bridge.get_clock().now().nanoseconds*1e-9-before >= self.cfg['step_seconds'],
                       timeout=5.0, residual=physical)
            self.last_action = action if self.residual_enabled else np.zeros(2)
            observation, info = self._fresh_observation(residual=physical)
            now = self.bridge.get_clock().now().nanoseconds*1e-9
            dt = max(now-before, 1e-6)
            terminated = self.result_future.done()
            truncated = not terminated and now-self.start_sim >= self.cfg['max_episode_seconds']
            outcome = None
            if terminated:
                result = self.result_future.result()
                outcome = navigation_outcome(result.status, result.result.error_code, result.result.error_msg)
                info['reason'] = result.result.error_msg
            elif truncated:
                outcome = 'TIME_LIMIT'
                self._cancel()
            reward, terms = calculate_reward(
                self.previous_progress, info['progress'], info['cross_track'], info['heading_error'],
                np.asarray(info['acceleration']), self.last_action, previous_action, dt, outcome, self.weights)
            self.previous_progress = info['progress']
            info.update(outcome=outcome, reward_terms=terms, simulation_time=now-self.start_sim,
                        step_duration=dt,
                        action=self.last_action.tolist(), commanded_residual=physical.tolist())
            self.records.write(json.dumps(info)+'\n')
            self.records.flush()
            self.finished = terminated or truncated
            if self.finished:
                self.bridge.command([0, 0])
            return observation, float(reward), terminated, truncated, info
        except RosWaitTimeout as exc:
            self._cleanup()
            raise InfrastructureFault(f'ROS data/clock timeout during episode: {exc}') from exc
        except BaseException:
            self._cleanup()
            raise

    def _cancel(self):
        if not self.context.ok():
            return
        if self.bridge:
            self.bridge.command([0, 0])
        if self.goal_handle and self.result_future and not self.result_future.done():
            try:
                self._result(self.goal_handle.cancel_goal_async(), timeout=2.0)
                self._result(self.result_future, timeout=7.0)
            except Exception:
                pass  # Cleanup also terminates our processes; controller watchdog remains active.

    def _cleanup(self):
        self._cancel()
        for child in reversed(self.children):
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGINT)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait(timeout=5)
        self.children.clear()
        for handle in self.log_handles:
            handle.close()
        self.log_handles.clear()
        if self.records:
            self.records.close()
            self.records = None
        if self.bridge:
            self.executor.shutdown(timeout_sec=2.0)
            if self.executor_thread is not None:
                self.executor_thread.join(timeout=2.0)
                if self.executor_thread.is_alive():
                    raise RuntimeError('ROS callback thread did not stop')
            self.executor.remove_node(self.bridge)
            self.bridge.destroy_node()
            self.bridge = None
            self.executor_thread = None
            self.executor_error = None
            self.executor = SingleThreadedExecutor(context=self.context)
        self.goal_handle = self.result_future = None

    def close(self):
        if not self.closed:
            self._cleanup()
            self.executor.shutdown()
            if self.context.ok():
                self.context.shutdown()
            self.closed = True

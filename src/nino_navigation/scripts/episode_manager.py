#!/usr/bin/env python3
"""One plan / one FollowPath per episode, with a latched velocity gate.

State/action callbacks run on a single-threaded executor. The execute coroutine yields
on an rclpy Future so monitoring/cancel callbacks continue without blocking.
An independent odometry reader timestamps delivery without changing task state.
Wall time supervises action waits and sensor delivery, even if /clock freezes.
"""

from copy import deepcopy
import hashlib
import json
from math import hypot, isfinite
from time import monotonic
from threading import Lock, Thread

import rclpy
from action_msgs.msg import GoalStatus
from action_msgs.srv import CancelGoal
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import ComputePathToPose, FollowPath, NavigateToPose
from nav_msgs.msg import Odometry, Path
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.serialization import serialize_message
from rclpy.task import Future
from rclpy.time import Time
from std_msgs.msg import Float64, String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener

from path_deviation import cross_track_error
from odom_warmup import OdomWarmup

INFRASTRUCTURE_ERROR = 65001


class PoseUnavailableError(ValueError):
    pass


class EpisodeManager(Node):
    def __init__(self):
        super().__init__('episode_manager')
        defaults = dict(max_path_deviation=0.20, monitor_rate=20.0, pose_timeout=0.5,
                        odom_warmup_duration=1.0,
                        candidate_timeout=0.5, planning_timeout=15.0, episode_timeout=180.0,
                        stop_timeout=5.0, stop_linear_tolerance=0.02,
                        stop_angular_tolerance=0.05, global_frame='map', base_frame='base_footprint')
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.cfg = {name: self.get_parameter(name).value for name in defaults}
        for name, value in self.cfg.items():
            if isinstance(value, (int, float)) and (not isfinite(value) or value <= 0):
                raise ValueError(f'{name} must be finite and positive')
        self.group = ReentrantCallbackGroup()
        self.odom_warmup = OdomWarmup(self.cfg['odom_warmup_duration'], self.cfg['pose_timeout']/2)
        self.failure_code = 1
        self.tf = Buffer()
        self.listener = TransformListener(self.tf, self)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.output = self.create_publisher(Twist, '/cmd_vel', 1)
        self.state_pub = self.create_publisher(String, '/episode/state', latched)
        self.reference_pub = self.create_publisher(Path, '/episode/reference_path', latched)
        self.error_pub = self.create_publisher(Float64, '/episode/path_deviation', 1)
        self.create_subscription(Twist, '/episode/cmd_vel_candidate', self.on_candidate, 1)
        self.create_subscription(PoseStamped, '/goal_pose', self.on_rviz_goal, 1)
        self.create_service(Trigger, '/episode/reset', self.reset)
        self.create_service(Trigger, '/episode/ready', self.ready)
        self.create_service(Trigger, '/episode/cancel', self.cancel_service)
        self.planner = ActionClient(self, ComputePathToPose, '/compute_path_to_pose', callback_group=self.group)
        self.follower = ActionClient(self, FollowPath, '/follow_path', callback_group=self.group)
        self.navigator = ActionServer(
            self, NavigateToPose, '/navigate_to_pose', self.execute,
            goal_callback=self.accept, cancel_callback=self.cancel, callback_group=self.group)
        self.rviz_client = ActionClient(self, NavigateToPose, '/navigate_to_pose', callback_group=self.group)
        self.cancel_client = self.create_client(CancelGoal, '/navigate_to_pose/_action/cancel_goal')
        self.state = 'IDLE'
        self.reason = ''
        self.episode_id = 0
        self.planning_requests = 0
        self.reference = None
        self.points = ()
        self.path_hash = ''
        self.busy = False
        self.child = None
        self.child_pending = False
        self.cancel_sent = False
        self.outer = None
        self.done = None
        self.odom = None
        self.odom_received = self.candidate_received = 0.0
        self._odom_lock = Lock()
        self._latest_odom = None
        self._consumed_odom = None
        # A slow planning callback must not prevent timestamping odometry
        # delivery. The reader ONLY caches messages; state changes and stop
        # commands remain on this node's single-threaded executor.
        self._odom_reader = Node(self.get_name()+'_odom_reader', context=self.context,
                                 use_global_arguments=False)
        self._odom_reader.create_subscription(Odometry, '/odom', self._cache_odometry, 1)
        self._odom_executor = SingleThreadedExecutor(context=self.context)
        self._odom_executor.add_node(self._odom_reader)
        self._odom_thread = Thread(target=self._spin_odometry, daemon=True)
        self._odom_thread.start()
        self.candidate = Twist()
        self.still_since = None
        self.set_state('IDLE')
        self.timer = self.create_timer(1.0/self.cfg['monitor_rate'], self.tick,
                                       clock=Clock(clock_type=ClockType.STEADY_TIME))

    def set_state(self, state, reason=''):
        self.state, self.reason = state, reason
        message = dict(episode_id=self.episode_id, state=state, reason=reason,
                       planning_requests=self.planning_requests, path_sha256=self.path_hash)
        self.state_pub.publish(String(data=json.dumps(message)))
        self.get_logger().info(json.dumps(message))

    def on_candidate(self, message):
        if all(isfinite(v) for v in (message.linear.x, message.linear.y, message.angular.z)):
            self.candidate, self.candidate_received = message, monotonic()

    def _cache_odometry(self, message):
        with self._odom_lock:
            self._latest_odom = (message, monotonic())

    def _spin_odometry(self):
        try:
            self._odom_executor.spin()
        except ExternalShutdownException:
            pass

    def _sync_odometry(self):
        with self._odom_lock:
            latest = self._latest_odom
        if latest is not None and latest is not self._consumed_odom:
            self._consumed_odom = latest
            self.on_odom(*latest)

    def on_odom(self, message, received_at=None):
        v = message.twist.twist
        if not all(isfinite(x) for x in (v.linear.x, v.linear.y, v.angular.z)):
            self.stop('FAILED', 'Non-finite odometry velocity', INFRASTRUCTURE_ERROR)
            return
        now = monotonic() if received_at is None else received_at
        stamp = message.header.stamp.sec*1000000000+message.header.stamp.nanosec
        if self.odom_warmup.stamp is not None and stamp < self.odom_warmup.stamp:
            self.stop('FAILED', 'Odometry clock moved backwards', INFRASTRUCTURE_ERROR)
        if self.odom_warmup.observe(now, stamp):
            self.odom, self.odom_received = message, now

    def stopped(self):
        self._sync_odometry()
        if self.odom is None or monotonic()-self.odom_received > self.cfg['pose_timeout']:
            return False
        v = self.odom.twist.twist
        return (hypot(v.linear.x, v.linear.y) <= self.cfg['stop_linear_tolerance']
                and abs(v.angular.z) <= self.cfg['stop_angular_tolerance'])

    def position(self):
        transform = self.tf.lookup_transform(self.cfg['global_frame'], self.cfg['base_frame'], Time())
        stamp = Time.from_msg(transform.header.stamp).nanoseconds
        age = (self.get_clock().now().nanoseconds-stamp)*1e-9
        if age < -0.1 or age > self.cfg['pose_timeout']:
            raise PoseUnavailableError(f'Stale or future robot TF: age={age:.3f}s')
        p = transform.transform.translation
        if not all(isfinite(v) for v in (p.x, p.y)):
            raise PoseUnavailableError('Non-finite robot TF')
        return transform

    def accept(self, request):
        if self.busy or self.state != 'IDLE' or request.behavior_tree:
            return GoalResponse.REJECT
        pose = request.pose
        q = pose.pose.orientation
        if (pose.header.frame_id != self.cfg['global_frame']
                or not all(isfinite(v) for v in (pose.pose.position.x, pose.pose.position.y, q.x, q.y, q.z, q.w))
                or abs(q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w-1.0) > 0.01):
            return GoalResponse.REJECT
        reason = self.readiness_reason()
        if reason:
            self.get_logger().warning(f'Goal rejected: {reason}')
            return GoalResponse.REJECT
        self.busy = True  # Reserve before the execute callback can yield.
        return GoalResponse.ACCEPT

    def readiness_reason(self):
        if self.busy or self.state != 'IDLE':
            return 'Episode is not IDLE'
        if not self.planner.server_is_ready() or not self.follower.server_is_ready():
            return 'Planner/controller action servers not ready'
        if not self.stopped():
            return 'Odometry not fresh or robot not stopped'
        if not self.odom_warmup.ready(monotonic()):
            return 'Waiting for continuous advancing odometry stream'
        if self.count_publishers('/cmd_vel') != 1:
            return 'Another /cmd_vel publisher exists'
        try:
            self.position()
        except (TransformException, ValueError) as exc:
            return str(exc)
        return ''

    def ready(self, request, response):
        reason = self.readiness_reason()
        response.success = not reason
        response.message = reason or 'Ready: fresh pose/odometry, stopped, action servers available'
        return response

    async def execute(self, handle):
        self.outer, self.done = handle, Future()
        self.episode_id += 1
        self.planning_requests = 1
        self.failure_code = 1
        self.reference, self.points, self.path_hash = None, (), ''
        self.candidate_received = 0.0
        self.started = monotonic()
        self.set_state('PLANNING')
        goal = ComputePathToPose.Goal()
        goal.goal, goal.planner_id = handle.request.pose, 'GridBased'
        goal.use_start = False
        self.send_child(self.planner, goal, 'PLANNING')
        outcome, reason = await self.done
        result = NavigateToPose.Result()
        result.error_code = 0 if outcome == 'SUCCEEDED' else self.failure_code
        result.error_msg = reason
        if outcome == 'SUCCEEDED':
            handle.succeed()
        elif outcome == 'CANCELED' and handle.is_cancel_requested:
            handle.canceled()
        else:
            handle.abort()
        self.outer = None
        return result

    def send_child(self, client, goal, phase):
        self.child_pending, self.cancel_sent = True, False
        future = client.send_goal_async(goal)
        future.add_done_callback(lambda f: self.child_accepted(f, phase))

    def child_accepted(self, future, phase):
        try:
            self.child = future.result()
            if not self.child.accepted:
                self.child_pending = False
                self.stop('FAILED', f'{phase} action rejected')
                return
            self.child.get_result_async().add_done_callback(lambda f: self.child_result(f, phase))
            if self.state != phase:
                self.cancel_child()  # Includes late acceptance after cancellation/timeout.
        except Exception as exc:
            self.child_pending = False
            self.stop('FAILED', f'{phase} action error: {exc}')

    def child_result(self, future, phase):
        self.child_pending, self.child = False, None
        try:
            wrapped = future.result()
            if self.state != phase:
                return
            if wrapped.status != GoalStatus.STATUS_SUCCEEDED or wrapped.result.error_code != 0:
                self.stop('FAILED', f'{phase}: {wrapped.result.error_msg or wrapped.status}')
                return
            if phase == 'PLANNING':
                path = deepcopy(wrapped.result.path)
                if path.header.frame_id != self.cfg['global_frame'] or not path.poses:
                    raise ValueError('Planner returned empty path or wrong frame')
                if any(p.header.frame_id not in ('', path.header.frame_id) for p in path.poses):
                    raise ValueError('Mixed frames in reference path')
                self.points = tuple((p.pose.position.x, p.pose.position.y) for p in path.poses)
                cross_track_error(*self.points[0], self.points)  # Validate finite coordinates.
                self.reference = path
                self.path_hash = hashlib.sha256(serialize_message(path)).hexdigest()
                self.reference_pub.publish(path)
                goal = FollowPath.Goal()
                goal.path = deepcopy(path)
                goal.controller_id, goal.goal_checker_id = 'FollowPath', 'general_goal_checker'
                goal.progress_checker_id = 'progress_checker'
                self.check_deviation()  # Never start following outside the allowed corridor.
                self.set_state('FOLLOWING')
                self.send_child(self.follower, goal, 'FOLLOWING')
            else:
                self.check_deviation()  # Deviation failure takes priority over success.
                self.stop('SUCCEEDED', 'Original path completed')
        except (TransformException, PoseUnavailableError) as exc:
            self.stop('FAILED', str(exc), INFRASTRUCTURE_ERROR)
        except Exception as exc:
            self.stop('FAILED', str(exc))

    def check_deviation(self):
        transform = self.position()
        p = transform.transform.translation
        error = cross_track_error(p.x, p.y, self.points)
        self.error_pub.publish(Float64(data=error))
        if error > self.cfg['max_path_deviation']:
            raise ValueError(f'Path deviation {error:.3f} m > {self.cfg["max_path_deviation"]:.3f} m')
        return transform

    def cancel_child(self):
        if self.child and self.child.accepted and not self.cancel_sent:
            self.cancel_sent = True
            self.child.cancel_goal_async()

    def stop(self, outcome, reason, error_code=1):
        if self.state not in ('PLANNING', 'FOLLOWING'):
            return
        self.outcome = outcome
        self.failure_code = error_code
        self.stop_started, self.still_since = monotonic(), None
        self.set_state('STOPPING', reason)
        self.output.publish(Twist())
        self.cancel_child()

    def cancel(self, handle):
        # A cancel request can arrive before execute() initializes the child.
        self.stop('CANCELED', 'Canceled by user')
        return CancelResponse.ACCEPT

    def cancel_service(self, request, response):
        response.success = self.state in ('PLANNING', 'FOLLOWING')
        if response.success:
            self.stop('CANCELED', 'Canceled via /episode/cancel')
            cancel = CancelGoal.Request()
            cancel.goal_info.goal_id = self.outer.goal_id
            self.cancel_client.call_async(cancel)
        response.message = self.state
        return response

    def reset(self, request, response):
        response.success = (self.state in ('SUCCEEDED', 'FAILED', 'CANCELED')
                            and not self.child_pending and self.outer is None and self.stopped())
        if response.success:
            self.busy = False
            self.set_state('IDLE', 'Reset; robot pose and map unchanged')
        response.message = self.state if response.success else 'Reset refused: action or robot not stopped'
        return response

    def on_rviz_goal(self, pose):
        goal = NavigateToPose.Goal()
        goal.pose = pose
        if self.rviz_client.server_is_ready():
            self.rviz_client.send_goal_async(goal).add_done_callback(self.rviz_accepted)

    def rviz_accepted(self, future):
        if not future.result().accepted:
            self.get_logger().warning('RViz goal rejected; check readiness and /episode/reset')

    def tick(self):
        self._sync_odometry()
        output = Twist()
        now = monotonic()
        if self.state in ('PLANNING', 'FOLLOWING'):
            if self.outer and self.outer.is_cancel_requested:
                self.stop('CANCELED', 'Canceled by user')
            elif now-self.started > self.cfg['episode_timeout']:
                self.stop('FAILED', 'Episode timeout')
            elif self.state == 'PLANNING' and now-self.started > self.cfg['planning_timeout']:
                self.stop('FAILED', 'Planning timeout; no retry')
            elif now-self.odom_received > self.cfg['pose_timeout']:
                self.stop('FAILED', 'Odometry delivery timeout', INFRASTRUCTURE_ERROR)
        if self.state == 'FOLLOWING':
            try:
                transform = self.check_deviation()
                if now-self.candidate_received <= self.cfg['candidate_timeout']:
                    output = self.candidate
                feedback = NavigateToPose.Feedback()
                feedback.current_pose.header = transform.header
                feedback.current_pose.pose.position.x = transform.transform.translation.x
                feedback.current_pose.pose.position.y = transform.transform.translation.y
                feedback.current_pose.pose.orientation = transform.transform.rotation
                feedback.distance_remaining = hypot(
                    self.points[-1][0]-transform.transform.translation.x,
                    self.points[-1][1]-transform.transform.translation.y)
                self.outer.publish_feedback(feedback)
            except (TransformException, PoseUnavailableError) as exc:
                self.stop('FAILED', str(exc), INFRASTRUCTURE_ERROR)
                output = Twist()
            except ValueError as exc:
                self.stop('FAILED', str(exc))
                output = Twist()
        self.output.publish(output)
        if self.state == 'STOPPING':
            if self.stopped():
                self.still_since = self.still_since or now
            else:
                self.still_since = None
            settled = self.still_since is not None and now-self.still_since >= 0.2
            if settled and not self.child_pending:
                self.finish(self.outcome, self.reason)
            elif now-self.stop_started > self.cfg['stop_timeout']:
                self.finish('FAILED', self.reason + '; stop/action completion not confirmed')

    def finish(self, outcome, reason):
        self.set_state(outcome, reason)
        if self.done is not None and not self.done.done():
            self.done.set_result((outcome, reason))

    def destroy_node(self):
        self._odom_executor.shutdown(timeout_sec=2.0)
        self._odom_thread.join(timeout=2.0)
        self._odom_reader.destroy_node()
        return super().destroy_node()


def main():
    rclpy.init()
    node = EpisodeManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.output.publish(Twist())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

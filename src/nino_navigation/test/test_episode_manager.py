"""In-process ROS action servers: count actual requests and inject failures.

Run without Gazebo/Nav2 on an unused ROS_DOMAIN_ID. No physics is simulated.
"""
from copy import deepcopy
from pathlib import Path as FilePath
import sys
from time import monotonic
from threading import Event, Thread
import time

import pytest
import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
from nav2_msgs.action import ComputePathToPose, FollowPath, NavigateToPose
from nav_msgs.msg import Odometry, Path
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.serialization import serialize_message
from rclpy.task import Future
from std_srvs.srv import Trigger
from tf2_ros import TransformBroadcaster

sys.path.insert(0, str(FilePath(__file__).resolve().parents[1] / 'scripts'))
from episode_manager import EpisodeManager


class Servers(Node):
    def __init__(self):
        super().__init__('test_nav2_servers')
        self.group = ReentrantCallbackGroup()
        self.plans = self.follows = 0
        self.y = 0.0
        self.publish_tf = True
        self.publish_odom = True
        self.hold_plan = False
        self.plan_release = Future()
        self.delay_follow_accept = False
        self.accept_release = Future()
        self.follow_release = Future()
        self.follow_handle = None
        self.fail_follow = False
        self.empty_plan = False
        self.received_path = None
        self.path = Path()
        self.path.header.frame_id = 'map'
        for x in (0.0, 1.0, 2.0):
            p = PoseStamped()
            p.header.frame_id = 'map'
            p.pose.position.x = x
            p.pose.orientation.w = 1.0
            self.path.poses.append(p)
        self.tf_pub = TransformBroadcaster(self)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 1)
        self.candidate_pub = self.create_publisher(Twist, '/episode/cmd_vel_candidate', 1)
        self.velocities = []
        self.create_subscription(Twist, '/cmd_vel', lambda m: self.velocities.append(m.linear.x), 100)
        self.timer = self.create_timer(0.02, self.tick)
        self.plan_server = ActionServer(self, ComputePathToPose, '/compute_path_to_pose',
                                       self.plan, callback_group=self.group,
                                       cancel_callback=lambda h: CancelResponse.ACCEPT)
        self.follow_server = ActionServer(self, FollowPath, '/follow_path', self.follow,
                                         callback_group=self.group, goal_callback=self.accept_follow,
                                         cancel_callback=lambda h: CancelResponse.ACCEPT)

    async def accept_follow(self, goal):
        self.follows += 1
        self.received_path = deepcopy(goal.path)
        if self.delay_follow_accept:
            await self.accept_release
        return GoalResponse.ACCEPT

    async def plan(self, handle):
        self.plans += 1
        if self.hold_plan:
            await self.plan_release
        result = ComputePathToPose.Result()
        if handle.is_cancel_requested:
            handle.canceled()
        else:
            handle.succeed()
            result.path = Path() if self.empty_plan else deepcopy(self.path)
        return result

    async def follow(self, handle):
        self.follow_handle = handle
        await self.follow_release
        result = FollowPath.Result()
        if handle.is_cancel_requested:
            handle.canceled()
        elif self.fail_follow:
            result.error_code = FollowPath.Result.NO_VALID_CONTROL
            result.error_msg = 'Injected obstruction'
            handle.abort()
        else:
            handle.succeed()
        return result

    def tick(self):
        if self.publish_tf:
            t = TransformStamped()
            t.header.frame_id, t.child_frame_id = 'map', 'base_footprint'
            t.header.stamp = self.get_clock().now().to_msg()
            t.transform.translation.y, t.transform.rotation.w = self.y, 1.0
            self.tf_pub.sendTransform(t)
        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        if self.publish_odom:
            self.odom_pub.publish(odom)
        candidate = Twist()
        candidate.linear.x = 0.15
        self.candidate_pub.publish(candidate)
        if self.follow_handle and self.follow_handle.is_cancel_requested and not self.follow_release.done():
            self.follow_release.set_result(None)


class Rig:
    def __init__(self):
        rclpy.init(domain_id=83)
        self.manager, self.servers = EpisodeManager(), Servers()
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.manager)
        self.executor.add_node(self.servers)
        self.client = ActionClient(self.servers, NavigateToPose, '/navigate_to_pose',
                                   callback_group=self.servers.group)
        self.wait(lambda: self.manager.stopped() and self.manager.tf.can_transform('map', 'base_footprint', rclpy.time.Time()))
        self.wait(lambda: self.manager.planner.server_is_ready() and self.manager.follower.server_is_ready() and self.client.server_is_ready())

    def wait(self, predicate, timeout=8):
        deadline = monotonic()+timeout
        while not predicate():
            assert monotonic() < deadline, f'Timed out: state={self.manager.state}, reason={self.manager.reason}'
            self.executor.spin_once(timeout_sec=0.02)

    def future(self, future):
        self.wait(future.done)
        return future.result()

    def start(self):
        self.wait(lambda: not self.manager.readiness_reason())
        goal = NavigateToPose.Goal()
        goal.pose = deepcopy(self.servers.path.poses[-1])
        handle = self.future(self.client.send_goal_async(goal))
        assert handle.accepted
        return handle

    def following(self):
        handle = self.start()
        self.wait(lambda: self.servers.follow_handle is not None)
        self.wait(lambda: 0.15 in self.servers.velocities)
        return handle

    def close(self):
        self.executor.remove_node(self.manager)
        self.executor.remove_node(self.servers)
        self.manager.destroy_node()
        self.servers.destroy_node()
        self.executor.shutdown()
        rclpy.shutdown()


@pytest.fixture
def rig():
    test = Rig()
    try:
        yield test
    finally:
        test.close()


def test_one_plan_original_path_and_success(rig):
    handle = rig.following()
    original = serialize_message(rig.manager.reference)
    assert original == serialize_message(rig.servers.received_path)
    rig.servers.y = 0.1  # Robot deviates but path must not be regenerated.
    start = monotonic()
    rig.wait(lambda: monotonic()-start > 1.2)
    assert rig.servers.plans == rig.servers.follows == 1
    assert serialize_message(rig.manager.reference) == original
    rig.servers.follow_release.set_result(None)
    result = rig.future(handle.get_result_async())
    assert result.status == GoalStatus.STATUS_SUCCEEDED
    assert rig.manager.state == 'SUCCEEDED'
    assert rig.servers.plans == 1


def test_deviation_aborts_latches_zero_and_reset(rig):
    handle = rig.following()
    rig.servers.y = 0.25
    rig.wait(lambda: rig.manager.state == 'STOPPING')
    count = len(rig.servers.velocities)
    result = rig.future(handle.get_result_async())
    assert result.status == GoalStatus.STATUS_ABORTED
    assert 'Path deviation' in result.result.error_msg
    assert rig.servers.plans == rig.servers.follows == 1
    assert all(v == 0 for v in rig.servers.velocities[count:])
    assert not rig.future(rig.client.send_goal_async(NavigateToPose.Goal())).accepted
    reset = rig.servers.create_client(Trigger, '/episode/reset')
    rig.wait(reset.service_is_ready)
    assert rig.future(reset.call_async(Trigger.Request())).success
    assert rig.manager.state == 'IDLE'


def test_controller_failure_never_replans(rig):
    handle = rig.following()
    rig.servers.fail_follow = True
    rig.servers.follow_release.set_result(None)
    result = rig.future(handle.get_result_async())
    assert result.status == GoalStatus.STATUS_ABORTED
    assert rig.servers.plans == rig.servers.follows == 1


def test_cancel_while_following(rig):
    handle = rig.following()
    assert rig.future(handle.cancel_goal_async()).goals_canceling
    result = rig.future(handle.get_result_async())
    assert result.status == GoalStatus.STATUS_CANCELED
    assert rig.servers.plans == 1


def test_cancel_with_late_follow_acceptance(rig):
    rig.servers.delay_follow_accept = True
    handle = rig.start()
    rig.wait(lambda: rig.servers.follows == 1)
    rig.future(handle.cancel_goal_async())
    rig.servers.accept_release.set_result(None)
    assert rig.future(handle.get_result_async()).status == GoalStatus.STATUS_CANCELED
    assert rig.servers.plans == rig.servers.follows == 1
    assert 0.15 not in rig.servers.velocities


def test_empty_plan_fails_without_follow(rig):
    rig.servers.empty_plan = True
    handle = rig.start()
    assert rig.future(handle.get_result_async()).status == GoalStatus.STATUS_ABORTED
    assert rig.servers.plans == 1
    assert rig.servers.follows == 0


def test_stale_tf_aborts(rig):
    handle = rig.following()
    rig.servers.publish_tf = False
    result = rig.future(handle.get_result_async())
    assert result.status == GoalStatus.STATUS_ABORTED
    assert 'TF' in result.result.error_msg


def test_cancel_during_planning(rig):
    rig.servers.hold_plan = True
    handle = rig.start()
    rig.wait(lambda: rig.servers.plans == 1)
    rig.future(handle.cancel_goal_async())
    rig.servers.plan_release.set_result(None)
    assert rig.future(handle.get_result_async()).status == GoalStatus.STATUS_CANCELED
    assert rig.servers.plans == 1 and rig.servers.follows == 0


def test_second_goal_rejected_during_episode(rig):
    handle = rig.following()
    goal = NavigateToPose.Goal()
    goal.pose = deepcopy(rig.servers.path.poses[-1])
    assert not rig.future(rig.client.send_goal_async(goal)).accepted
    assert rig.servers.plans == rig.servers.follows == 1
    rig.future(handle.cancel_goal_async())
    rig.future(handle.get_result_async())


def test_cancel_service_reports_canceled(rig):
    handle = rig.following()
    cancel = rig.servers.create_client(Trigger, '/episode/cancel')
    rig.wait(cancel.service_is_ready)
    assert rig.future(cancel.call_async(Trigger.Request())).success
    assert rig.future(handle.get_result_async()).status == GoalStatus.STATUS_CANCELED


def test_timeout_keeps_gate_closed_until_late_acceptance(rig):
    rig.servers.delay_follow_accept = True
    rig.manager.cfg['stop_timeout'] = 0.3
    handle = rig.start()
    rig.wait(lambda: rig.servers.follows == 1)
    rig.future(handle.cancel_goal_async())
    assert rig.future(handle.get_result_async()).status == GoalStatus.STATUS_ABORTED
    reset = rig.servers.create_client(Trigger, '/episode/reset')
    rig.wait(reset.service_is_ready)
    assert not rig.future(reset.call_async(Trigger.Request())).success
    rig.servers.accept_release.set_result(None)
    rig.wait(lambda: not rig.manager.child_pending)
    assert 0.15 not in rig.servers.velocities
    assert rig.future(reset.call_async(Trigger.Request())).success


def test_planning_timeout_has_no_retry_or_follow(rig):
    rig.servers.hold_plan = True
    rig.manager.cfg['planning_timeout'] = 0.15
    rig.manager.cfg['stop_timeout'] = 0.3
    handle = rig.start()
    assert rig.future(handle.get_result_async()).status == GoalStatus.STATUS_ABORTED
    rig.servers.plan_release.set_result(None)
    rig.wait(lambda: not rig.manager.child_pending)
    assert rig.servers.plans == 1 and rig.servers.follows == 0


def test_ready_service_uses_managers_own_tf_and_rejects_busy(rig):
    ready = rig.servers.create_client(Trigger, '/episode/ready')
    rig.wait(ready.service_is_ready)
    rig.wait(lambda: not rig.manager.readiness_reason())
    assert rig.future(ready.call_async(Trigger.Request())).success
    rig.servers.publish_tf = False
    start = monotonic()
    rig.wait(lambda: monotonic()-start > 0.6)
    reply = rig.future(ready.call_async(Trigger.Request()))
    assert not reply.success
    assert 'TF' in reply.message
    assert rig.servers.plans == 0
    rig.servers.publish_tf = True
    rig.wait(lambda: not rig.manager.readiness_reason())
    handle = rig.following()
    assert not rig.future(ready.call_async(Trigger.Request())).success
    rig.future(handle.cancel_goal_async())
    rig.future(handle.get_result_async())


def test_real_odom_loss_still_aborts_and_commands_zero(rig):
    handle = rig.following()
    rig.servers.publish_odom = False
    result = rig.future(handle.get_result_async())
    assert result.status == GoalStatus.STATUS_ABORTED
    assert result.result.error_code == 65001
    assert 'Odometry delivery timeout' in result.result.error_msg
    assert rig.servers.plans == 1
    assert all(v == 0 for v in rig.servers.velocities[-5:])


def test_busy_main_executor_does_not_fake_odom_loss(rig):
    stop = Event()

    def independent_publisher():
        while not stop.is_set():
            message = Odometry()
            message.header.stamp = rig.servers.get_clock().now().to_msg()
            rig.servers.odom_pub.publish(message)
            stop.wait(0.02)

    publisher = Thread(target=independent_publisher)
    publisher.start()
    try:
        # Model a long planning/serialization callback without spinning the
        # main executor. Real odometry continues to arrive independently.
        time.sleep(0.7)
        rig.manager._sync_odometry()
        assert monotonic()-rig.manager.odom_received < 0.2
        assert rig.manager.stopped()
    finally:
        stop.set()
        publisher.join(timeout=2)

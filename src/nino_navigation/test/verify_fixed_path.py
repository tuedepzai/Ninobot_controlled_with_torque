#!/usr/bin/env python3
"""Gazebo integration check. Run alone: starts/stops its own simulation.

Usage: source install/setup.bash; python3 src/nino_navigation/test/verify_fixed_path.py
Writes a fresh temporary map and logs under log/fixed_path_<timestamp>/.
"""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
from time import monotonic

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from lifecycle_msgs.srv import GetState
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry, Path as PathMessage
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.serialization import serialize_message
from std_msgs.msg import String
from std_srvs.srv import Trigger


class Probe(Node):
    def __init__(self):
        super().__init__('verify_fixed_path')
        self.states, self.paths, self.velocities = [], [], []
        self.odom = None
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, '/episode/state', lambda m: self.states.append(json.loads(m.data)), latched)
        self.create_subscription(PathMessage, '/episode/reference_path',
                                 lambda m: self.paths.append(serialize_message(m)), latched)
        self.create_subscription(Twist, '/cmd_vel',
                                 lambda m: self.velocities.append((monotonic(), m.linear.x, m.angular.z)), 100)
        self.create_subscription(Odometry, '/odom', self.on_odom, 10)
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 1)
        self.client = ActionClient(self, NavigateToPose, '/navigate_to_pose')
        self.reset_client = self.create_client(Trigger, '/episode/reset')

    def on_odom(self, message):
        self.odom = message

    def wait(self, predicate, timeout=60):
        end = monotonic()+timeout
        while not predicate():
            if monotonic() > end:
                raise RuntimeError(f'Timed out. Last state: {self.states[-1:]}')
            rclpy.spin_once(self, timeout_sec=0.05)

    def result(self, future, timeout=60):
        self.wait(future.done, timeout)
        return future.result()

    def settle(self, seconds):
        start = monotonic()
        self.wait(lambda: monotonic()-start > seconds)

    def initialize_pose(self, x=0.0, y=0.0):
        self.wait(lambda: self.pose_pub.get_subscription_count() > 0)
        pose = PoseWithCovarianceStamped()
        pose.header.frame_id = 'map'
        pose.pose.pose.position.x = x
        pose.pose.pose.position.y = y
        pose.pose.pose.orientation.w = 1.0
        pose.pose.covariance[0] = pose.pose.covariance[7] = 0.01
        pose.pose.covariance[35] = 0.01
        self.pose_pub.publish(pose)

    def goal(self, x):
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'
        goal.pose.pose.position.x = x
        goal.pose.pose.orientation.w = 1.0
        handle = self.result(self.client.send_goal_async(goal))
        assert handle.accepted, 'Episode rejected; see navigation.log'
        return handle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--standard', action='store_true', help='Regression: original Nav2 launch')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    logs = root/'log'/('fixed_path_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    logs.mkdir(parents=True)
    children = []

    def start(name, command):
        child = subprocess.Popen(command, stdout=(logs/(name+'.log')).open('w'),
                                 stderr=subprocess.STDOUT, start_new_session=True)
        children.append(child)
        return child

    def stop(child):
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGINT)
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=10)

    def run(command, timeout=90):
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        (logs/'commands.log').open('a').write(f'{command}\n{result.stdout}\n{result.stderr}\n')
        result.check_returncode()
        return result.stdout

    print(f'Logs: {logs}', flush=True)
    rclpy.init()
    probe = Probe()
    try:
        # Refuse to run alongside a different simulation or navigation session.
        probe.settle(0.5)
        assert probe.count_publishers('/clock') == 0, 'Stop existing Gazebo before testing'
        start('sim', ['ros2', 'launch', 'nino_description', 'sim.launch.py', 'headless:=true'])
        mapping = start('mapping', ['ros2', 'launch', 'nino_navigation', 'mapping.launch.py', 'rviz:=false'])
        run(['ros2', 'topic', 'echo', '/map', 'nav_msgs/msg/OccupancyGrid', '--once', '--field', 'info'])
        run(['ros2', 'run', 'nav2_map_server', 'map_saver_cli', '-f', str(logs/'map'),
             '--ros-args', '-p', 'use_sim_time:=true', '-p', 'save_map_timeout:=15.0'])
        stop(mapping)
        start('navigation', ['ros2', 'launch', 'nino_navigation', 'navigation.launch.py',
                             'rviz:=false', 'fixed_path:='+('false' if args.standard else 'true'),
                             'map:='+str(logs/'map.yaml')])
        probe.wait(probe.client.server_is_ready)
        probe.initialize_pose()
        last_server = '/bt_navigator' if args.standard else '/collision_monitor'
        state_client = probe.create_client(GetState, last_server+'/get_state')
        probe.wait(state_client.service_is_ready)
        for _ in range(60):
            active = probe.result(state_client.call_async(GetState.Request())).current_state.id
            if active == 3:
                break
            probe.settle(0.5)
        assert active == 3
        probe.settle(1.0)
        handle = probe.goal(1.0)
        result = probe.result(handle.get_result_async(), 90)
        assert result.status == GoalStatus.STATUS_SUCCEEDED, str(result)
        print('Real Gazebo/Nav2: 1 m goal SUCCEEDED', flush=True)
        if args.standard:
            (logs/'result.json').write_text(json.dumps({'standard_navigation_success': True}))
            return
        assert probe.states[-1]['state'] == 'SUCCEEDED'
        assert probe.states[-1]['planning_requests'] == 1
        assert len(set(probe.paths)) == 1
        assert probe.count_publishers('/cmd_vel') == 1
        probe.wait(lambda: probe.odom is not None)
        assert abs(probe.odom.twist.twist.linear.x) <= 0.02
        probe.wait(probe.reset_client.service_is_ready)
        assert probe.result(probe.reset_client.call_async(Trigger.Request())).success
        before = len(probe.paths)
        handle = probe.goal(3.0)
        probe.wait(lambda: probe.states[-1]['state'] == 'FOLLOWING')
        probe.wait(lambda: len(probe.paths) > before)
        frozen = probe.paths[-1]
        # Move the Gazebo robot laterally; wheel odometry cannot measure this
        # displacement, so AMCL must detect the shifted lidar scan.
        x = probe.odom.pose.pose.position.x
        reply = run(['gz', 'service', '-s', '/world/long_hall/set_pose',
                     '--reqtype', 'gz.msgs.Pose', '--reptype', 'gz.msgs.Boolean',
                     '--timeout', '5000', '--req',
                     f'name: "nino", position: {{x: {x}, y: 0.65, z: 0.02}}, orientation: {{w: 1.0}}'])
        assert 'true' in reply
        # Make the known relocation visible to the estimator as well. AMCL
        # alone is not guaranteed to recover a large unmodelled lateral jump.
        probe.initialize_pose(x, 0.65)
        result = probe.result(handle.get_result_async(), 90)
        assert result.status == GoalStatus.STATUS_ABORTED, str(result)
        assert 'Path deviation' in result.result.error_msg, result.result.error_msg
        assert len(probe.paths) == before+1 and probe.paths[-1] == frozen
        assert probe.states[-1]['planning_requests'] == 1
        probe.settle(0.5)
        assert all(abs(v) < 1e-9 and abs(w) < 1e-9 for _,v,w in probe.velocities[-5:])
        assert abs(probe.odom.twist.twist.linear.x) <= 0.02
        assert abs(probe.odom.twist.twist.angular.z) <= 0.05
        evidence = dict(success=True, deviation_failure=True, one_plan_per_episode=True,
                        reference_sha256=hashlib.sha256(frozen).hexdigest(),
                        states=probe.states, final_linear_speed=probe.odom.twist.twist.linear.x,
                        final_angular_speed=probe.odom.twist.twist.angular.z)
        (logs/'result.json').write_text(json.dumps(evidence, indent=2))
        print('Real Gazebo/Nav2: lateral displacement -> FAILED; unchanged reference; stopped', flush=True)
    finally:
        for child in reversed(children):
            stop(child)
        probe.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

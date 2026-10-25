"""Exercise the actual torque adapter over ROS topics, without motors/Gazebo."""

import json
from time import monotonic

import rclpy
from geometry_msgs.msg import Twist
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String

from nino_control.effort_drive import EffortDrive


def test_residual_adapter_gates_stop_failure_and_timeout():
    rclpy.init(args=['--ros-args', '-p', 'enable_residual:=true'], domain_id=84)
    adapter, probe = EffortDrive(), Node('residual_test_probe')
    executor = SingleThreadedExecutor()
    executor.add_node(adapter)
    executor.add_node(probe)
    state = probe.create_publisher(String, '/episode/state',
                                  QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    cmd = probe.create_publisher(Twist, '/cmd_vel', 1)
    joints = probe.create_publisher(JointState, '/joint_states', 1)
    residual = probe.create_publisher(Float64MultiArray, '/wheel_torque_residual', 1)
    received = []
    probe.create_subscription(Float64MultiArray, '/wheel_residual_applied',
                              lambda m: received.append(list(m.data)), 10)

    def spin(seconds):
        deadline = monotonic()+seconds
        while monotonic() < deadline:
            executor.spin_once(timeout_sec=0.005)

    def publish(linear, following, values, repeat=True):
        state.publish(String(data=json.dumps({'state': 'FOLLOWING' if following else 'FAILED'})))
        spin(0.03)
        joint = JointState(name=['left_wheel_joint', 'right_wheel_joint'], velocity=[0.0, 0.0])
        command = Twist()
        command.linear.x = linear
        deadline = monotonic()+0.15
        while monotonic() < deadline:
            joints.publish(joint)
            cmd.publish(command)
            if repeat:
                residual.publish(Float64MultiArray(data=values))
            spin(0.01)

    try:
        spin(0.2)
        publish(0.2, True, [0.3, -0.3])
        assert any(abs(v[0]) > 0.05 for v in received)
        publish(0.0, True, [0.3, 0.3])
        assert received[-1] == [0.0, 0.0]
        publish(0.2, False, [0.3, 0.3])
        assert received[-1] == [0.0, 0.0]
        publish(0.2, True, [0.3, 0.3])
        assert received[-1][0] > 0
        spin(0.25)
        assert received[-1] == [0.0, 0.0]
    finally:
        executor.remove_node(adapter)
        executor.remove_node(probe)
        adapter.destroy_node()
        probe.destroy_node()
        executor.shutdown()
        rclpy.shutdown()

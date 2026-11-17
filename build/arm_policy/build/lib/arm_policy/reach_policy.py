import os
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Point
from tf2_ros import Buffer, TransformListener
from stable_baselines3 import SAC
from visualization_msgs.msg import Marker


JOINTS = [f"fr3_joint{i}" for i in range(1, 8)]
HOME = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])   # MuJoCo's home pose
LOW = np.array([-2.9007, -1.8361, -2.9007, -3.077, -2.8763, 0.4398, -3.0508])  # from explore_model.py
HIGH = np.array([2.9007, 1.8361, 2.9007, -0.1169, 2.8763, 4.6216, 3.0508])
MAX_DELTA = 0.05   # same as training


class ReachPolicy(Node):
    def __init__(self):
        super().__init__("reach_policy")
        self.declare_parameter("model_path", os.path.expanduser("~/Projects/RoboArmSim/rl/fr3_reach_sac.zip"))
        self.model = SAC.load(self.get_parameter("model_path").value, device="cpu")

        self.q = None            # measured joint angles (like data.qpos)
        self.target = None       # where to reach
        self.q_cmd = HOME.copy() # commanded angles (like data.ctrl)
        self.reached = False

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(JointState, "/joint_states", self.joint_cb, 10)
        self.create_subscription(Point, "/reach_target", self.target_cb, 10)
        self.cmd_pub = self.create_publisher(Float64MultiArray, "/arm_controller/commands", 10)
        self.create_timer(0.05, self.control_loop)   # 20 Hz, same as training
        self.get_logger().info("Policy loaded. Publish a target on /reach_target")
        self.marker_pub = self.create_publisher(Marker, "/target_marker", 10)

    def joint_cb(self, msg):
        # /joint_states doesn't guarantee joint order, so match by name
        pos = dict(zip(msg.name, msg.position))
        if all(j in pos for j in JOINTS):
            self.q = np.array([pos[j] for j in JOINTS])

    def target_cb(self, msg):
        self.target = np.array([msg.x, msg.y, msg.z])
        self.reached = False
        self.get_logger().info(f"New target: {self.target}")

        m = Marker()
        m.header.frame_id = "fr3_link0"
        m.type = Marker.SPHERE
        m.pose.position = msg
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.05   # 5 cm, the "reached" distance
        m.color.r, m.color.a = 1.0, 0.8           # red, slightly see-through
        self.marker_pub.publish(m)

    def get_flange(self):
        try:
            t = self.tf_buffer.lookup_transform("fr3_link0", "fr3_link8", Time())
        except Exception:
            return None
        p = t.transform.translation
        return np.array([p.x, p.y, p.z])

    def publish(self, q):
        self.cmd_pub.publish(Float64MultiArray(data=q.tolist()))

    def control_loop(self):
        flange = self.get_flange()
        if self.q is None or flange is None:
            return
        if self.target is None:
            self.publish(self.q_cmd)   # hold home until a target arrives
            return

        # Same observation as fr3_reach_env._get_obs()
        obs = np.concatenate([self.q, flange, self.target, self.target - flange]).astype(np.float32)
        action, _ = self.model.predict(obs, deterministic=True)

        new_cmd = np.clip(self.q_cmd + action * MAX_DELTA, LOW, HIGH)
        self.q_cmd = 0.7 * self.q_cmd + 0.3 * new_cmd
        self.publish(self.q_cmd)

        distance = np.linalg.norm(self.target - flange)
        if distance < 0.05 and not self.reached:
            self.reached = True
            self.get_logger().info(f"Reached! distance {distance:.3f} m")


def main():
    rclpy.init()
    rclpy.spin(ReachPolicy())

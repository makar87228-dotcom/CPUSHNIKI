#!/usr/bin/env python3
import math
import os
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import Twist
from unitree_go.msg import LowCmd, LowState

STAND_Q = np.array([0.0, 0.8536, -1.7072] * 4, dtype=np.float64)
JOINT_MIN = np.array([-0.863, -0.686, -2.818] * 4, dtype=np.float64)
JOINT_MAX = np.array([0.863, 4.501, -0.888] * 4, dtype=np.float64)

ACTION_SCALE = 0.35
KP = 35.0
KD = 1.0
GAIT_HZ = 1.5
POLICY_RATE_HZ = 50.0


class NumpyPolicy:
    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        self.layers = [
            (data["mlp_extractor.policy_net.0.weight"], data["mlp_extractor.policy_net.0.bias"]),
            (data["mlp_extractor.policy_net.2.weight"], data["mlp_extractor.policy_net.2.bias"]),
            (data["mlp_extractor.policy_net.4.weight"], data["mlp_extractor.policy_net.4.bias"]),
        ]
        self.out_w = data["action_net.weight"]
        self.out_b = data["action_net.bias"]

    def predict(self, obs: np.ndarray) -> np.ndarray:
        x = obs.astype(np.float32)
        for w, b in self.layers:
            x = np.tanh(x @ w.T + b)
        return np.clip(x @ self.out_w.T + self.out_b, -1.0, 1.0)


def gravity_body_from_quat(quat_wxyz) -> np.ndarray:
    """Поворачивает мировой вектор [0, 0, -1] в систему корпуса (w, x, y, z)."""
    w, x, y, z = (float(v) for v in quat_wxyz)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm > 1e-6:
        w, x, y, z = w / norm, x / norm, y / norm, z / norm
    else:
        w, x, y, z = 1.0, 0.0, 0.0, 0.0
    return np.array(
        [
            2.0 * (w * y - x * z),
            -2.0 * (w * x + y * z),
            2.0 * (x * x + y * y) - 1.0,
        ],
        dtype=np.float64,
    )


class Go1RLPolicyNode(Node):
    def __init__(self):
        super().__init__(
            "go1_rl_policy",
            parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)],
        )

        base_dir = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(base_dir, "policy_weights.npz")
        self.policy = NumpyPolicy(weights_path)

        self.latest_state: LowState | None = None
        self.cmd_vx = 0.0
        self.last_action = np.zeros(12, dtype=np.float64)
        self.phase = 0.0
        self.prev_time_s: float | None = None

        qos_state = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        qos_cmd = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.sub_state = self.create_subscription(
            LowState, "/lowstate", self.on_lowstate, qos_state
        )
        self.sub_vel = self.create_subscription(
            Twist, "/cmd_vel", self.on_cmd_vel, qos_state
        )
        self.pub_cmd = self.create_publisher(LowCmd, "/lowcmd_bare", qos_cmd)

        self.timer = self.create_timer(1.0 / POLICY_RATE_HZ, self.on_timer)
        self.get_logger().info("Go1 RL Policy node ready (50 Hz, use_sim_time=True)")

    def on_lowstate(self, msg: LowState):
        self.latest_state = msg

    def on_cmd_vel(self, msg: Twist):
        vx = float(msg.linear.x)
        if math.isfinite(vx):
            self.cmd_vx = min(max(vx, 0.0), 0.60)

    def on_timer(self):
        now_ns = self.get_clock().now().nanoseconds
        if now_ns <= 0 or self.latest_state is None:
            return
        now_s = now_ns * 1e-9

        if self.prev_time_s is None:
            dt = 0.0
        else:
            dt = max(0.0, min(0.05, now_s - self.prev_time_s))
        self.prev_time_s = now_s

        st = self.latest_state
        grav_body = gravity_body_from_quat(st.imu_state.quaternion)
        gyro = np.array(st.imu_state.gyroscope, dtype=np.float64)
        q = np.array([float(st.motor_state[i].q) for i in range(12)], dtype=np.float64)
        dq = np.array([float(st.motor_state[i].dq) for i in range(12)], dtype=np.float64)

        if not (
            np.all(np.isfinite(grav_body))
            and np.all(np.isfinite(gyro))
            and np.all(np.isfinite(q))
            and np.all(np.isfinite(dq))
        ):
            return

        phase_obs = (
            np.array(
                [math.sin(2.0 * math.pi * self.phase), math.cos(2.0 * math.pi * self.phase)],
                dtype=np.float64,
            )
            if self.cmd_vx > 0.02
            else np.array([0.0, 1.0], dtype=np.float64)
        )

        obs = np.concatenate([
            grav_body,
            gyro * 0.25,
            q - STAND_Q,
            dq * 0.05,
            self.last_action,
            np.array([self.cmd_vx * 2.0, 0.0, 0.0], dtype=np.float64),
            phase_obs,
        ]).astype(np.float32)

        action = self.policy.predict(obs).astype(np.float64)
        q_target = np.clip(STAND_Q + action * ACTION_SCALE, JOINT_MIN, JOINT_MAX)

        out = LowCmd()
        out.head = [0xFE, 0xEF]
        out.level_flag = 0xFF
        for i in range(12):
            m = out.motor_cmd[i]
            m.mode = 0x0A
            m.q = float(q_target[i])
            m.dq = 0.0
            m.tau = 0.0
            m.kp = KP
            m.kd = KD

        self.pub_cmd.publish(out)
        self.last_action = action
        if self.cmd_vx > 0.02:
            self.phase = (self.phase + GAIT_HZ * dt) % 1.0
        else:
            self.phase = 0.0


def main(args=None):
    rclpy.init(args=args)
    node = Go1RLPolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

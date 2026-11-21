#!/usr/bin/env python3
import csv
import math
import os
import numpy as np
import mujoco
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from unitree_go.msg import LowCmd, LowState

JOINT_LIMITS = [
    (-0.863, 0.863),   # 0: FR_hip
    (-0.686, 4.501),   # 1: FR_thigh
    (-2.818, -0.888),  # 2: FR_calf
    (-0.863, 0.863),   # 3: FL_hip
    (-0.686, 4.501),   # 4: FL_thigh
    (-2.818, -0.888),  # 5: FL_calf
    (-0.863, 0.863),   # 6: RR_hip
    (-0.686, 4.501),   # 7: RR_thigh
    (-2.818, -0.888),  # 8: RR_calf
    (-0.863, 0.863),   # 9: RL_hip
    (-0.686, 4.501),   # 10: RL_thigh
    (-2.818, -0.888),  # 11: RL_calf
]

KP_MIN, KP_MAX = 0.0, 1000.0
KD_MIN, KD_MAX = 0.0, 100.0
TAU_MIN, TAU_MAX = -100.0, 100.0
DQ_MIN, DQ_MAX = -20.0, 20.0

MAX_STEP_RAD = math.radians(5.0) - 1e-6
CHECK_SUBSTEP_RAD = math.radians(0.1)
PENETRATION_LIMIT = -0.001  # строго более 1 мм -> dist < -0.001

HOME_Q = np.array([
    0.0, 0.9, -1.8,
    0.0, 0.9, -1.8,
    0.0, 0.9, -1.8,
    0.0, 0.9, -1.8,
], dtype=np.float64)

HOLD_KP = 40.0
HOLD_KD = 2.0
INPUT_TIMEOUT_SEC = 0.5
PUBLISH_RATE_HZ = 100.0


def load_checked_pairs(csv_path: str) -> set:
    pairs = set()
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('diagnostic_policy', '').strip() == 'checked':
                g1 = int(row['geom1_id'])
                g2 = int(row['geom2_id'])
                pairs.add((min(g1, g2), max(g1, g2)))
    return pairs


class Go1SafetyFilter(Node):
    def __init__(self):
        super().__init__(
            'go1_safety_filter',
            parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL, True)]
        )

        base_dir = os.path.dirname(os.path.abspath(__file__))
        scene_path = os.path.join(base_dir, 'model', 'scene.xml')
        csv_path = os.path.join(base_dir, 'model', 't2_collision_pairs.csv')

        self.checked_pairs = load_checked_pairs(csv_path)
        self.mj_model = mujoco.MjModel.from_xml_path(scene_path)
        self.mj_data = mujoco.MjData(self.mj_model)

        self.q_safe = HOME_Q.copy()
        self.dq_safe = np.zeros(12, dtype=np.float64)
        self.tau_safe = np.zeros(12, dtype=np.float64)
        self.kp_safe = np.full(12, HOLD_KP, dtype=np.float64)
        self.kd_safe = np.full(12, HOLD_KD, dtype=np.float64)
        self.mode_safe = [0x0A] * 12

        self.initialized_from_state = False
        self.has_received_cmd = False
        self.last_valid_cmd_time = None

        qos_cmd = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE
        )
        qos_state = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE
        )

        self.sub_state = self.create_subscription(LowState, '/lowstate', self.on_lowstate, qos_state)
        self.sub_cmd = self.create_subscription(LowCmd, '/lowcmd_bare', self.on_lowcmd_bare, qos_cmd)
        self.pub_cmd = self.create_publisher(LowCmd, '/lowcmd', qos_cmd)

        self.timer = self.create_timer(1.0 / PUBLISH_RATE_HZ, self.on_timer)
        self.get_logger().info(f'Go1 Safety Filter ready (100 Hz, {len(self.checked_pairs)} collision pairs)')

    def has_forbidden_collision(self, q_vec: np.ndarray) -> bool:
        self.mj_data.qpos[0:3] = [0.0, 0.0, 2.0]
        self.mj_data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.mj_data.qpos[7:19] = q_vec
        mujoco.mj_forward(self.mj_model, self.mj_data)

        for i in range(self.mj_data.ncon):
            c = self.mj_data.contact[i]
            if c.dist < PENETRATION_LIMIT:
                pair = (min(int(c.geom1), int(c.geom2)), max(int(c.geom1), int(c.geom2)))
                if pair in self.checked_pairs:
                    return True
        return False

    def is_transition_safe(self, q_start: np.ndarray, q_end: np.ndarray) -> bool:
        max_diff = float(np.max(np.abs(q_end - q_start)))
        steps = max(1, int(math.ceil(max_diff / CHECK_SUBSTEP_RAD)))
        for s in range(1, steps + 1):
            alpha = s / float(steps)
            q_interp = (1.0 - alpha) * q_start + alpha * q_end
            if self.has_forbidden_collision(q_interp):
                return False
        return True

    def on_lowstate(self, msg: LowState):
        if self.initialized_from_state or self.has_received_cmd:
            return
        q_cur = np.zeros(12, dtype=np.float64)
        for i in range(12):
            val = float(msg.motor_state[i].q)
            if not math.isfinite(val):
                return
            lo, hi = JOINT_LIMITS[i]
            q_cur[i] = min(max(val, lo), hi)

        if not self.has_forbidden_collision(q_cur):
            self.q_safe = q_cur
        self.initialized_from_state = True

    def on_lowcmd_bare(self, msg: LowCmd):
        q_raw = np.zeros(12, dtype=np.float64)
        dq_raw = np.zeros(12, dtype=np.float64)
        tau_raw = np.zeros(12, dtype=np.float64)
        kp_raw = np.zeros(12, dtype=np.float64)
        kd_raw = np.zeros(12, dtype=np.float64)
        modes = [0x0A] * 12

        for i in range(12):
            m = msg.motor_cmd[i]
            q_v, dq_v, tau_v, kp_v, kd_v = (
                float(m.q), float(m.dq), float(m.tau), float(m.kp), float(m.kd)
            )
            if not all(math.isfinite(v) for v in (q_v, dq_v, tau_v, kp_v, kd_v)):
                return

            lo, hi = JOINT_LIMITS[i]
            if not (lo - 1e-4 <= q_v <= hi + 1e-4):
                return
            if not (DQ_MIN <= dq_v <= DQ_MAX):
                return
            if not (TAU_MIN <= tau_v <= TAU_MAX):
                return
            if not (KP_MIN <= kp_v <= KP_MAX):
                return
            if not (KD_MIN <= kd_v <= KD_MAX):
                return

            q_raw[i] = min(max(q_v, lo), hi)
            dq_raw[i] = dq_v
            tau_raw[i] = tau_v
            kp_raw[i] = kp_v
            kd_raw[i] = kd_v
            modes[i] = int(m.mode)

        delta = np.clip(q_raw - self.q_safe, -MAX_STEP_RAD, MAX_STEP_RAD)
        q_cand = self.q_safe + delta

        for i in range(12):
            lo, hi = JOINT_LIMITS[i]
            q_cand[i] = min(max(q_cand[i], lo), hi)

        if not self.is_transition_safe(self.q_safe, q_cand):
            return

        self.q_safe = q_cand
        self.dq_safe = dq_raw
        self.tau_safe = tau_raw
        self.kp_safe = kp_raw
        self.kd_safe = kd_raw
        self.mode_safe = modes
        self.has_received_cmd = True
        self.last_valid_cmd_time = self.get_clock().now()

    def on_timer(self):
        now = self.get_clock().now()
        if now.nanoseconds == 0:
            return

        in_hold = True
        if self.has_received_cmd and self.last_valid_cmd_time is not None:
            dt = (now - self.last_valid_cmd_time).nanoseconds * 1e-9
            if dt <= INPUT_TIMEOUT_SEC:
                in_hold = False

        out = LowCmd()
        out.head = [0xFE, 0xEF]
        out.level_flag = 0xFF

        for i in range(12):
            cmd = out.motor_cmd[i]
            cmd.q = float(self.q_safe[i])
            if in_hold:
                cmd.mode = 0x0A
                cmd.dq = 0.0
                cmd.tau = 0.0
                cmd.kp = float(HOLD_KP)
                cmd.kd = float(HOLD_KD)
            else:
                cmd.mode = self.mode_safe[i]
                cmd.dq = float(self.dq_safe[i])
                cmd.tau = float(self.tau_safe[i])
                cmd.kp = float(self.kp_safe[i])
                cmd.kd = float(self.kd_safe[i])

        for i in range(12, 20):
            cmd = out.motor_cmd[i]
            cmd.mode = 0
            cmd.q = 0.0
            cmd.dq = 0.0
            cmd.tau = 0.0
            cmd.kp = 0.0
            cmd.kd = 0.0

        self.pub_cmd.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = Go1SafetyFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

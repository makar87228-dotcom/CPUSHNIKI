import math
import os
import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

LEG_ORDER = ["FR", "FL", "RR", "RL"]
JOINT_KINDS = ["hip", "thigh", "calf"]
JOINT_NAMES = [f"{leg}_{kind}_joint" for leg in LEG_ORDER for kind in JOINT_KINDS]

STAND_Q = np.array([0.0, 0.8536, -1.7072] * 4, dtype=np.float64)
TORQUE_LIMIT = np.array([23.7, 23.7, 35.55] * 4, dtype=np.float64)

JOINT_MIN = np.array([-0.863, -0.686, -2.818] * 4, dtype=np.float64)
JOINT_MAX = np.array([0.863, 4.501, -0.888] * 4, dtype=np.float64)

MAX_STEP_RAD = math.radians(5.0) - 1e-6
ACTION_SCALE = 0.35
KP = 35.0
KD = 1.0
GAIT_HZ = 1.5

SAFETY_SUBSTEPS = 2
PHYS_PER_SAFETY = 5


def quat_to_rpy(w: float, x: float, y: float, z: float) -> tuple[float, float, float]:
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(sinp)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


class Go1WalkEnv(gym.Env):
    """Точная среда MuJoCo для T3 (соответствует go1_mujoco.py + core.py + safety_filter.py)."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, eval_mode: bool = False, fixed_cmd_vx: float | None = None):
        super().__init__()
        self.eval_mode = eval_mode
        self.fixed_cmd_vx = fixed_cmd_vx

        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        scene_path = os.path.join(root_dir, "model", "scene.xml")

        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)

        self.model.actuator_gainprm[:, 0] = 0.0
        self.model.actuator_biasprm[:, :] = 0.0

        self.u_qadr = []
        self.u_dadr = []
        for name in JOINT_NAMES:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.u_qadr.append(int(self.model.jnt_qposadr[jid]))
            self.u_dadr.append(int(self.model.jnt_dofadr[jid]))

        self.floor_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.foot_geoms = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, leg): idx
            for idx, leg in enumerate(LEG_ORDER)
        }
        trunk_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        self.trunk_geoms = {
            g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] == trunk_bid
        }

        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(47,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(12,), dtype=np.float32
        )

        self.max_steps = 600
        self.step_count = 0
        self.phase = 0.0
        self.warmup_steps = 0
        self.target_vx = 0.35
        self.cmd_vx = 0.35
        self.q_safe = STAND_Q.copy()
        self.last_action = np.zeros(12, dtype=np.float64)
        self.start_x = 0.0
        self.start_y = 0.0
        self.rng = np.random.default_rng()

    def _get_q_dq(self) -> tuple[np.ndarray, np.ndarray]:
        q = np.array([self.data.qpos[a] for a in self.u_qadr], dtype=np.float64)
        dq = np.array([self.data.qvel[a] for a in self.u_dadr], dtype=np.float64)
        return q, dq

    def _get_obs(self) -> np.ndarray:
        q, dq = self._get_q_dq()
        inv_quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_negQuat(inv_quat, self.data.qpos[3:7])

        grav_body = np.zeros(3, dtype=np.float64)
        mujoco.mju_rotVecQuat(
            grav_body, np.array([0.0, 0.0, -1.0], dtype=np.float64), inv_quat
        )
        gyro = self.data.qvel[3:6].copy()

        if not self.eval_mode:
            grav_body += self.rng.normal(0.0, 0.015, size=3)
            gyro += self.rng.normal(0.0, 0.03, size=3)
            q = q + self.rng.normal(0.0, 0.01, size=12)
            dq = dq + self.rng.normal(0.0, 0.05, size=12)

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
        ])
        return obs.astype(np.float32)

    def _contacts_info(self) -> tuple[list[bool], bool, bool]:
        foot_contacts = [False, False, False, False]
        bad_contact = False
        trunk_hit = False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 == self.floor_geom or g2 == self.floor_geom:
                other = g2 if g1 == self.floor_geom else g1
                if other in self.foot_geoms:
                    foot_contacts[self.foot_geoms[other]] = True
                else:
                    bad_contact = True
                    if other in self.trunk_geoms:
                        trunk_hit = True
            else:
                if c.dist < -0.002:
                    bad_contact = True
        return foot_contacts, bad_contact, trunk_hit

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:3] = [0.0, 0.0, 0.30]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

        for idx, a in enumerate(self.u_qadr):
            noise = 0.0 if self.eval_mode else float(self.rng.uniform(-0.04, 0.04))
            self.data.qpos[a] = STAND_Q[idx] + noise
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self.q_safe = STAND_Q.copy()
        for _ in range(50):
            q, dq = self._get_q_dq()
            tau = np.clip(40.0 * (STAND_Q - q) + 2.0 * (0.0 - dq), -TORQUE_LIMIT, TORQUE_LIMIT)
            self.data.qfrc_applied[:] = 0.0
            for k, a in enumerate(self.u_dadr):
                self.data.qfrc_applied[a] = tau[k]
            mujoco.mj_step(self.model, self.data)

        self.step_count = 0
        self.phase = 0.0
        self.last_action = np.zeros(12, dtype=np.float64)
        self.q_safe, _ = self._get_q_dq()
        self.q_safe = np.clip(self.q_safe, JOINT_MIN, JOINT_MAX)

        if self.fixed_cmd_vx is not None:
            self.target_vx = float(self.fixed_cmd_vx)
            self.warmup_steps = 75 if self.eval_mode else 0
        else:
            self.target_vx = float(self.rng.choice([0.30, 0.35, 0.40, 0.45]))
            self.warmup_steps = int(self.rng.integers(0, 75))

        self.cmd_vx = 0.0 if self.warmup_steps > 0 else self.target_vx
        self.start_x = float(self.data.qpos[0])
        self.start_y = float(self.data.qpos[1])

        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        q_target = np.clip(STAND_Q + action * ACTION_SCALE, JOINT_MIN, JOINT_MAX)

        if not self.eval_mode and self.rng.random() < 0.015:
            self.data.qvel[0] += float(self.rng.uniform(-0.12, 0.12))
            self.data.qvel[1] += float(self.rng.uniform(-0.10, 0.10))

        last_tau = np.zeros(12, dtype=np.float64)
        for _ in range(SAFETY_SUBSTEPS):
            delta = np.clip(q_target - self.q_safe, -MAX_STEP_RAD, MAX_STEP_RAD)
            self.q_safe = np.clip(self.q_safe + delta, JOINT_MIN, JOINT_MAX)
            for _ in range(PHYS_PER_SAFETY):
                q, dq = self._get_q_dq()
                tau = np.clip(
                    KP * (self.q_safe - q) + KD * (0.0 - dq),
                    -TORQUE_LIMIT,
                    TORQUE_LIMIT,
                )
                self.data.qfrc_applied[:] = 0.0
                for k, a in enumerate(self.u_dadr):
                    self.data.qfrc_applied[a] = tau[k]
                mujoco.mj_step(self.model, self.data)
                last_tau = tau

        self.step_count += 1
        if self.step_count >= self.warmup_steps:
            self.cmd_vx = self.target_vx
        if self.cmd_vx > 0.02:
            self.phase = (self.phase + GAIT_HZ * 0.02) % 1.0

        inv_quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_negQuat(inv_quat, self.data.qpos[3:7])
        v_local = np.zeros(3, dtype=np.float64)
        mujoco.mju_rotVecQuat(v_local, self.data.qvel[0:3], inv_quat)
        grav_body = np.zeros(3, dtype=np.float64)
        mujoco.mju_rotVecQuat(
            grav_body, np.array([0.0, 0.0, -1.0], dtype=np.float64), inv_quat
        )

        q, _ = self._get_q_dq()
        roll, pitch, yaw = quat_to_rpy(*self.data.qpos[3:7])
        z = float(self.data.qpos[2])
        foot_contacts, bad_contact, trunk_hit = self._contacts_info()

        vx = float(v_local[0])
        if self.cmd_vx > 0.02:
            rew_vel = (
                2.5 * math.exp(-((vx - self.cmd_vx) ** 2) / (0.25 ** 2))
                + 2.0 * min(vx, self.cmd_vx + 0.05)
            )
        else:
            rew_vel = 2.0 * math.exp(-(vx ** 2) / (0.12 ** 2))

        rew_lat = -1.5 * (float(v_local[1]) ** 2) - 0.8 * (float(self.data.qvel[5]) ** 2)
        rew_yaw = -0.3 * (yaw ** 2)
        rew_height = -18.0 * ((z - 0.295) ** 2)
        rew_orient = (
            -4.5 * (float(grav_body[0]) ** 2 + float(grav_body[1]) ** 2)
            - 0.08 * (float(self.data.qvel[3]) ** 2 + float(self.data.qvel[4]) ** 2)
        )
        rew_hip = -1.2 * float(np.sum(q[[0, 3, 6, 9]] ** 2))
        rew_pose = -0.15 * float(np.sum((q - STAND_Q) ** 2))
        rew_smooth = -0.06 * float(np.sum((action - self.last_action) ** 2))
        rew_torque = -0.0004 * float(np.sum(last_tau ** 2))

        rew_gait = 0.0
        if self.cmd_vx > 0.02:
            diag_a = foot_contacts[0] and foot_contacts[3] and (not foot_contacts[1]) and (not foot_contacts[2])
            diag_b = foot_contacts[1] and foot_contacts[2] and (not foot_contacts[0]) and (not foot_contacts[3])
            if self.phase < 0.5 and diag_a:
                rew_gait = 0.4
            elif self.phase >= 0.5 and diag_b:
                rew_gait = 0.4
            elif sum(foot_contacts) == 0:
                rew_gait = -0.5
        else:
            if sum(foot_contacts) == 4:
                rew_gait = 0.3

        rew_contact = -1.0 if bad_contact else 0.0
        rew_alive = 0.6

        reward = (
            rew_vel
            + rew_lat
            + rew_yaw
            + rew_height
            + rew_orient
            + rew_hip
            + rew_pose
            + rew_smooth
            + rew_torque
            + rew_gait
            + rew_contact
            + rew_alive
        )

        self.last_action = action.copy()

        fell = trunk_hit or z < 0.175 or abs(roll) > 0.72 or abs(pitch) > 0.72
        if fell:
            reward -= 15.0

        terminated = bool(fell)
        truncated = bool(self.step_count >= self.max_steps)

        info = {
            "dist_x": float(self.data.qpos[0]) - self.start_x,
            "fell": fell,
        }
        return self._get_obs(), float(reward), terminated, truncated, info


def heuristic_trot_action(phase: float, cmd_vx: float) -> np.ndarray:
    """Базовый шаблон диагональной рыси для Этапа 1 (Pretrain)."""
    if cmd_vx <= 0.02:
        return np.zeros(12, dtype=np.float32)

    scale_vx = min(1.3, max(0.3, cmd_vx / 0.35))
    dq = np.zeros(12, dtype=np.float64)
    phases = [phase, (phase + 0.5) % 1.0, (phase + 0.5) % 1.0, phase]
    for leg_idx, p in enumerate(phases):
        phi = 2.0 * math.pi * p
        lift = max(0.0, math.sin(phi))
        dq_thigh = 0.28 * lift + 0.14 * scale_vx * math.cos(phi)
        dq_calf = -0.56 * lift
        base = leg_idx * 3
        dq[base + 0] = 0.0
        dq[base + 1] = dq_thigh
        dq[base + 2] = dq_calf

    return np.clip(dq / ACTION_SCALE, -1.0, 1.0).astype(np.float32)

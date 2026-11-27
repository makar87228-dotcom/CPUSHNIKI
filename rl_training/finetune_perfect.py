import math
import os
import time
import numpy as np
import mujoco
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from go1_sim_env import (
    Go1WalkEnv, STAND_Q, TORQUE_LIMIT, JOINT_MIN, JOINT_MAX,
    MAX_STEP_RAD, ACTION_SCALE, KP, KD, GAIT_HZ,
    SAFETY_SUBSTEPS, PHYS_PER_SAFETY, quat_to_rpy
)
from train_rl import export_policy_npz


class Go1PrecisionEnv(Go1WalkEnv):
    """Строгое отслеживание заданной скорости (0.30..0.60) и идеально прямой курс."""

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        if self.fixed_cmd_vx is None:
            self.target_vx = float(self.rng.choice([0.30, 0.35, 0.45, 0.55, 0.60]))
            self.cmd_vx = 0.0 if self.warmup_steps > 0 else self.target_vx
        return self._get_obs(), info

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        q_target = np.clip(STAND_Q + action * ACTION_SCALE, JOINT_MIN, JOINT_MAX)

        if not self.eval_mode and self.rng.random() < 0.015:
            self.data.qvel[0] += float(self.rng.uniform(-0.10, 0.10))
            self.data.qvel[1] += float(self.rng.uniform(-0.08, 0.08))

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
        dy = float(self.data.qpos[1]) - self.start_y
        foot_contacts, bad_contact, trunk_hit = self._contacts_info()

        vx = float(v_local[0])
        vy = float(v_local[1])
        wz = float(self.data.qvel[5])

        # Узкая строгая награда за точное попадание в cmd_vx (без превышения!)
        if self.cmd_vx > 0.02:
            rew_vel = 3.5 * math.exp(-((vx - self.cmd_vx) ** 2) / (0.10 ** 2))
        else:
            rew_vel = 2.5 * math.exp(-(vx ** 2) / (0.08 ** 2))

        # Жёсткий контроль прямолинейности (без увода вбок и разворота)
        rew_straight = (
            -2.5 * (vy ** 2)
            - 3.0 * (wz ** 2)
            - 12.0 * (yaw ** 2)
            - 8.0 * (dy ** 2)
        )

        rew_height = -18.0 * ((z - 0.295) ** 2)
        rew_orient = (
            -5.0 * (float(grav_body[0]) ** 2 + float(grav_body[1]) ** 2)
            - 0.10 * (float(self.data.qvel[3]) ** 2 + float(self.data.qvel[4]) ** 2)
        )
        rew_hip = -2.0 * float(np.sum(q[[0, 3, 6, 9]] ** 2))
        rew_pose = -0.15 * float(np.sum((q - STAND_Q) ** 2))
        rew_smooth = -0.08 * float(np.sum((action - self.last_action) ** 2))
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
            rew_vel + rew_straight + rew_height + rew_orient
            + rew_hip + rew_pose + rew_smooth + rew_torque
            + rew_gait + rew_contact + rew_alive
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


def make_prec_env(rank: int, seed: int = 200):
    def _init():
        env = Go1PrecisionEnv(eval_mode=False)
        env.reset(seed=seed + rank)
        return env
    return _init


def eval_one_speed(model: PPO, cmd_vx: float) -> tuple[float, float, float, int]:
    env = Go1PrecisionEnv(eval_mode=True, fixed_cmd_vx=cmd_vx)
    env.max_steps = 600  # 1.5с прогрев + 10.5с движение
    vs, dys, yaws, falls = [], [], [], 0
    for ep in range(2):
        obs, _ = env.reset(seed=1100 + ep)
        dx_warm = 0.0
        fell = False
        for step in range(env.max_steps):
            act, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(act)
            if step == 74:
                dx_warm = float(env.data.qpos[0]) - env.start_x
            if info["fell"]:
                fell = True
            if term or trunc:
                break
        dx = float(env.data.qpos[0]) - env.start_x
        dy = abs(float(env.data.qpos[1]) - env.start_y)
        _, _, yaw = quat_to_rpy(*env.data.qpos[3:7])
        vs.append((dx - dx_warm) / 10.5)
        dys.append(dy)
        yaws.append(abs(math.degrees(yaw)))
        if fell:
            falls += 1
    return float(np.mean(vs)), float(np.mean(dys)), float(np.mean(yaws)), falls


class PrecisionEvalCallback(BaseCallback):
    def __init__(self, eval_freq: int, art_dir: str):
        super().__init__()
        self.eval_freq = eval_freq
        self.art_dir = art_dir
        self.best_err = 1e9

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            v35, dy35, yaw35, f35 = eval_one_speed(self.model, 0.35)
            v60, dy60, yaw60, f60 = eval_one_speed(self.model, 0.60)
            print(
                f"[Prec @ {self.num_timesteps:7d}] "
                f"0.35->v={v35:.2f}(|dy|={dy35:.2f},yaw={yaw35:.1f}°) | "
                f"0.60->v={v60:.2f}(|dy|={dy60:.2f},yaw={yaw60:.1f}°) | falls={f35+f60}/4"
            )
            err = abs(v35 - 0.35) + abs(v60 - 0.60) + 0.5 * (dy35 + dy60) + 0.02 * (yaw35 + yaw60)
            if (f35 + f60) == 0 and err < self.best_err:
                self.best_err = err
                self.model.save(os.path.join(self.art_dir, "best_walk_model.zip"))
                export_policy_npz(self.model, os.path.join(self.art_dir, "best_policy_weights.npz"))
        return True


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    art_dir = os.path.join(base_dir, "artifacts")
    n_envs = min(8, max(4, (os.cpu_count() or 8) - 2))

    vec_env = SubprocVecEnv([make_prec_env(i) for i in range(n_envs)])
    model = PPO.load(os.path.join(art_dir, "best_walk_model.zip"), env=vec_env, device="cpu")
    model.learning_rate = 1.5e-4

    cb = PrecisionEvalCallback(eval_freq=max(1, 30_000 // n_envs), art_dir=art_dir)
    print("=== Дообучение: точная скорость + идеально ровный курс ===")
    t0 = time.time()
    try:
        model.learn(total_timesteps=500_000, callback=cb, reset_num_timesteps=False, progress_bar=False)
    except KeyboardInterrupt:
        print("Остановлено пользователем.")
    finally:
        print(f"Время дообучения: {(time.time() - t0) / 60.0:.1f} мин")
        model.save(os.path.join(art_dir, "final_model.zip"))
        export_policy_npz(model, os.path.join(art_dir, "policy_weights.npz"))
        try:
            vec_env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()

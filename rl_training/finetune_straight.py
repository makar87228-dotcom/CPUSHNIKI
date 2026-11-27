import math
import os
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from go1_sim_env import Go1WalkEnv, quat_to_rpy
from train_rl import export_policy_npz


class Go1StraightFastEnv(Go1WalkEnv):
    """Дообучение: ровно прямой курс (без увода влево) и скорость до 0.60 м/с."""

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        if self.fixed_cmd_vx is None:
            self.target_vx = float(self.rng.choice([0.35, 0.50, 0.60]))
            self.cmd_vx = 0.0 if self.warmup_steps > 0 else self.target_vx
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        _, _, yaw = quat_to_rpy(*self.data.qpos[3:7])
        dy = float(self.data.qpos[1]) - self.start_y

        # Дополнительный штраф за увод влево/вправо и разворот корпуса
        straight_penalty = -1.8 * (yaw ** 2) - 2.0 * (dy ** 2)
        return obs, float(reward + straight_penalty), terminated, truncated, info


def make_straight_env(rank: int, seed: int = 100):
    def _init():
        env = Go1StraightFastEnv(eval_mode=False)
        env.reset(seed=seed + rank)
        return env
    return _init


class StraightEvalCallback(BaseCallback):
    def __init__(self, eval_freq: int, art_dir: str):
        super().__init__()
        self.eval_freq = eval_freq
        self.art_dir = art_dir
        self.best_score = -1e9

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            env = Go1StraightFastEnv(eval_mode=True, fixed_cmd_vx=0.60)
            dxs, dys, yaws, falls = [], [], [], 0
            for ep in range(3):
                obs, _ = env.reset(seed=900 + ep)
                fell = False
                for _ in range(600):
                    act, _ = self.model.predict(obs, deterministic=True)
                    obs, _, term, trunc, info = env.step(act)
                    if info["fell"]:
                        fell = True
                    if term or trunc:
                        break
                _, _, yaw = quat_to_rpy(*env.data.qpos[3:7])
                dxs.append(float(env.data.qpos[0]) - env.start_x)
                dys.append(abs(float(env.data.qpos[1]) - env.start_y))
                yaws.append(abs(math.degrees(yaw)))
                if fell:
                    falls += 1
            mdx, mdy, myaw = float(np.mean(dxs)), float(np.mean(dys)), float(np.mean(yaws))
            print(
                f"[FineTune @ {self.num_timesteps:7d}] dx={mdx:.2f}м | ср.|dy|={mdy:.2f}м | "
                f"ср.|yaw|={myaw:.1f}° | падений={falls}/3"
            )
            score = mdx - 2.0 * mdy - 0.1 * myaw
            if falls == 0 and score > self.best_score:
                self.best_score = score
                self.model.save(os.path.join(self.art_dir, "best_walk_model.zip"))
                export_policy_npz(self.model, os.path.join(self.art_dir, "best_policy_weights.npz"))
        return True


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    art_dir = os.path.join(base_dir, "artifacts")
    n_envs = min(8, max(4, (os.cpu_count() or 8) - 2))

    vec_env = SubprocVecEnv([make_straight_env(i) for i in range(n_envs)])
    model = PPO.load(os.path.join(art_dir, "best_walk_model.zip"), env=vec_env, device="cpu")
    model.learning_rate = 1e-4

    cb = StraightEvalCallback(eval_freq=max(1, 30_000 // n_envs), art_dir=art_dir)
    print("=== ЭТАП 3: Мягкое дообучение на прямой курс и скорость 0.60 м/с ===")
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

import os
import time
import math
import numpy as np
from mujoco import viewer as mujoco_viewer

from go1_sim_env import Go1WalkEnv, quat_to_rpy


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


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    art_dir = os.path.join(base_dir, "artifacts")
    best_npz = os.path.join(art_dir, "best_policy_weights.npz")
    final_npz = os.path.join(art_dir, "policy_weights.npz")
    npz_path = best_npz if os.path.isfile(best_npz) else final_npz
    print(f"Загружаю веса: {npz_path}")

    policy = NumpyPolicy(npz_path)
    env = Go1WalkEnv(eval_mode=True, fixed_cmd_vx=0.55)
    env.max_steps = 825  # 1.5 с прогрев + 15 с движение

    obs, _ = env.reset(seed=777)

    print("\n--- Старт 3D-просмотра (закройте окно MuJoCo после просмотра) ---")
    with mujoco_viewer.launch_passive(env.model, env.data) as v:
        v.cam.distance = 2.2
        v.cam.elevation = -20
        v.cam.azimuth = 90

        for step in range(env.max_steps):
            if not v.is_running():
                break
            t0 = time.perf_counter()

            act = policy.predict(obs)
            obs, _, term, trunc, info = env.step(act)

            # Камера плавно едет за корпусом
            v.cam.lookat[0] = env.data.qpos[0]
            v.cam.lookat[1] = env.data.qpos[1]
            v.cam.lookat[2] = 0.28
            v.sync()

            if step % 50 == 0 or term or trunc:
                roll, pitch, yaw = quat_to_rpy(*env.data.qpos[3:7])
                dx = env.data.qpos[0] - env.start_x
                dy = env.data.qpos[1] - env.start_y
                z = env.data.qpos[2]
                print(
                    f"[t={step * 0.02:5.2f}s | cmd={env.cmd_vx:.2f}] "
                    f"вперёд dx={dx:+.2f}м | вбок dy={dy:+.2f}м | "
                    f"yaw={math.degrees(yaw):+.1f}° | z={z:.2f}м | fell={info['fell']}"
                )

            if term or trunc:
                time.sleep(1.5)
                break

            dt_left = 0.02 - (time.perf_counter() - t0)
            if dt_left > 0:
                time.sleep(dt_left)


if __name__ == "__main__":
    main()

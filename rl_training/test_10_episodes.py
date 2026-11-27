import math
import os
import numpy as np

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


def run_eval(policy, cmd_vx: float, episodes: int = 10):
    env = Go1WalkEnv(eval_mode=False, fixed_cmd_vx=cmd_vx)
    env.max_steps = 825  # 1.5 с прогрев (75 шагов) + 15 с ход (750 шагов)
    passed = 0

    print(f"\n=== Проверка на скорости cmd_vx={cmd_vx:.2f} м/с ({episodes} эпизодов) ===")
    for ep in range(1, episodes + 1):
        obs, _ = env.reset(seed=3000 + ep)
        env.warmup_steps = 75
        env.cmd_vx = 0.0

        min_z = 9.9
        max_tilt = 0.0
        fell = False
        dx_after_warmup = 0.0

        for step in range(env.max_steps):
            act = policy.predict(obs)
            obs, _, term, trunc, info = env.step(act)
            if step == 74:
                dx_after_warmup = float(env.data.qpos[0]) - env.start_x
            roll, pitch, _ = quat_to_rpy(*env.data.qpos[3:7])
            min_z = min(min_z, float(env.data.qpos[2]))
            max_tilt = max(max_tilt, abs(math.degrees(roll)), abs(math.degrees(pitch)))
            if info["fell"]:
                fell = True
            if term or trunc:
                break

        dx = float(env.data.qpos[0]) - env.start_x
        dy = float(env.data.qpos[1]) - env.start_y
        _, _, yaw = quat_to_rpy(*env.data.qpos[3:7])
        real_speed = (dx - dx_after_warmup) / 15.0

        ok = (not fell) and dx >= 3.0
        if ok:
            passed += 1
        status = "PASS" if ok else "FAIL"
        print(
            f"Эп.{ep:2d} [{status}] | dx={dx:+.2f}м | v_факт={real_speed:.2f}м/с | "
            f"вбок dy={dy:+.2f}м | yaw={math.degrees(yaw):+.1f}° | "
            f"min_z={min_z:.2f}м | max_наклон={max_tilt:.1f}°"
        )

    print(f"ИТОГ ({cmd_vx:.2f} м/с): {passed}/{episodes} PASS")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    art_dir = os.path.join(base_dir, "artifacts")
    best_npz = os.path.join(art_dir, "best_policy_weights.npz")
    final_npz = os.path.join(art_dir, "policy_weights.npz")
    npz_path = best_npz if os.path.isfile(best_npz) else final_npz
    print(f"Веса: {npz_path}")

    policy = NumpyPolicy(npz_path)
    run_eval(policy, cmd_vx=0.55, episodes=10)
    run_eval(policy, cmd_vx=0.60, episodes=2)


if __name__ == "__main__":
    main()

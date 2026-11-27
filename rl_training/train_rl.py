import argparse
import csv
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv

from go1_sim_env import Go1WalkEnv, heuristic_trot_action


def make_env(rank: int, seed: int = 42):
    def _init():
        env = Go1WalkEnv(eval_mode=False)
        env.reset(seed=seed + rank)
        return env
    return _init


def export_policy_npz(model: PPO, out_path: str):
    """Сохраняет веса MLP-актора в лёгкий .npz для быстрого NumPy-инференса в Docker."""
    state_dict = model.policy.state_dict()
    arrays = {}
    for k, v in state_dict.items():
        if k.startswith("mlp_extractor.policy_net.") or k.startswith("action_net."):
            arrays[k] = v.detach().cpu().numpy().astype(np.float32)
    np.savez(out_path, **arrays)

    test_obs = np.random.randn(1, 47).astype(np.float32) * 0.2
    pt_act, _ = model.predict(test_obs, deterministic=True)
    x = test_obs[0]
    for idx in (0, 2, 4):
        w = arrays[f"mlp_extractor.policy_net.{idx}.weight"]
        b = arrays[f"mlp_extractor.policy_net.{idx}.bias"]
        x = np.tanh(x @ w.T + b)
    np_act = np.clip(x @ arrays["action_net.weight"].T + arrays["action_net.bias"], -1.0, 1.0)
    max_diff = float(np.max(np.abs(pt_act[0] - np_act)))
    print(f"[Export] Веса сохранены в {out_path} | Сверка PyTorch vs NumPy max_diff={max_diff:.2e}")


def pretrain_actor(model: PPO, device: str, epochs: int = 35):
    """Этап 1: быстрый Pretrain базовой диагональной рыси."""
    print("\n=== ЭТАП 1: Pretrain базовой рыси (warm-start) ===")
    env = Go1WalkEnv(eval_mode=True, fixed_cmd_vx=0.35)
    obs_list, act_list = [], []

    for ep in range(20):
        vx = 0.0 if ep < 3 else float(np.random.uniform(0.25, 0.48))
        env.fixed_cmd_vx = vx
        obs, _ = env.reset(seed=100 + ep)
        env.warmup_steps = 0
        env.cmd_vx = vx
        for _ in range(250):
            act = heuristic_trot_action(env.phase, env.cmd_vx)
            obs_list.append(obs.copy())
            act_list.append(act.copy())
            obs, _, term, trunc, _ = env.step(act)
            if term or trunc:
                break

    X = torch.tensor(np.array(obs_list), dtype=torch.float32, device=device)
    Y = torch.tensor(np.array(act_list), dtype=torch.float32, device=device)
    print(f"Собрано {len(X)} обучающих кадров базовой походки.")

    actor_params = list(model.policy.mlp_extractor.policy_net.parameters()) + list(
        model.policy.action_net.parameters()
    )
    opt = torch.optim.Adam(actor_params, lr=1e-3)
    loss_fn = nn.MSELoss()

    batch_size = 256
    for ep in range(1, epochs + 1):
        perm = torch.randperm(len(X), device=device)
        total_loss = 0.0
        n_batches = 0
        for i in range(0, len(X), batch_size):
            idx = perm[i : i + batch_size]
            feat = model.policy.mlp_extractor.policy_net(X[idx])
            pred = model.policy.action_net(feat)
            loss = loss_fn(pred, Y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
            n_batches += 1
        if ep % 10 == 0 or ep == epochs:
            print(f"  Pretrain эпоха {ep:2d}/{epochs} | MSE={total_loss / max(1, n_batches):.5f}")


class DistanceEvalCallback(BaseCallback):
    """Периодически замеряет реальную дистанцию и падения в eval-режиме."""

    def __init__(self, eval_freq: int, artifacts_dir: str):
        super().__init__()
        self.eval_freq = eval_freq
        self.artifacts_dir = artifacts_dir
        self.history = []
        self.best_dist = -1e9

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            env = Go1WalkEnv(eval_mode=True, fixed_cmd_vx=0.35)
            dists, rews, falls = [], [], 0
            for ep in range(3):
                obs, _ = env.reset(seed=500 + ep)
                ep_rew = 0.0
                fell = False
                dist_x = 0.0
                for _ in range(600):
                    act, _ = self.model.predict(obs, deterministic=True)
                    obs, r, term, trunc, info = env.step(act)
                    ep_rew += r
                    dist_x = info["dist_x"]
                    if info["fell"]:
                        fell = True
                    if term or trunc:
                        break
                dists.append(dist_x)
                rews.append(ep_rew)
                if fell:
                    falls += 1
            mean_d = float(np.mean(dists))
            mean_r = float(np.mean(rews))
            steps = self.num_timesteps
            print(
                f"[Eval @ {steps:8d} steps] ср. дистанция (за 10.5с ход) = {mean_d:.2f} м | "
                f"ср. награда = {mean_r:.1f} | падений = {falls}/3"
            )
            self.history.append((steps, mean_d, mean_r, falls))
            if falls == 0 and mean_d > self.best_dist:
                self.best_dist = mean_d
                self.model.save(os.path.join(self.artifacts_dir, "best_walk_model.zip"))
                export_policy_npz(
                    self.model, os.path.join(self.artifacts_dir, "best_policy_weights.npz")
                )
        return True


def save_plots_and_summary(artifacts_dir: str, eval_cb: DistanceEvalCallback):
    csv_path = os.path.join(artifacts_dir, "progress.csv")
    if os.path.isfile(csv_path):
        steps, rewards = [], []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("rollout/ep_rew_mean") and row.get("time/total_timesteps"):
                    steps.append(float(row["time/total_timesteps"]))
                    rewards.append(float(row["rollout/ep_rew_mean"]))
        if steps:
            fig, ax1 = plt.subplots(figsize=(9, 5))
            ax1.plot(steps, rewards, label="Mean Episode Reward (PPO)", color="tab:blue")
            ax1.set_xlabel("Timesteps")
            ax1.set_ylabel("Mean Reward", color="tab:blue")
            ax1.grid(True, alpha=0.3)
            if eval_cb.history:
                ax2 = ax1.twinx()
                es = [h[0] for h in eval_cb.history]
                ed = [h[1] for h in eval_cb.history]
                ax2.plot(es, ed, "o-", label="Eval Distance (m)", color="tab:green")
                ax2.set_ylabel("Walk Distance (m)", color="tab:green")
            plt.title("Unitree Go1 RL Fine-Tuning (MuJoCo + SafetyFilter)")
            fig.tight_layout()
            plt.savefig(os.path.join(artifacts_dir, "training_curve.png"), dpi=150)
            plt.close(fig)

    if eval_cb.history:
        eval_csv = os.path.join(artifacts_dir, "eval_history.csv")
        with open(eval_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["timesteps", "mean_distance_m", "mean_reward", "falls_out_of_3"])
            w.writerows(eval_cb.history)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=min(8, max(4, (os.cpu_count() or 8) - 2)))
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    artifacts_dir = os.path.join(base_dir, "artifacts")
    tb_dir = os.path.join(artifacts_dir, "tb_logs")
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)

    device = "cpu"
    print(f"Устройство обучения: {device} | Параллельных сред MuJoCo: {args.n_envs}")

    config = {
        "algorithm": "PPO (Pretrain + RL Fine-tuning)",
        "total_timesteps": args.steps,
        "n_envs": args.n_envs,
        "device": device,
        "policy_freq_hz": 50,
        "safety_filter_freq_hz": 100,
        "kp": 35.0,
        "kd": 1.0,
        "action_scale": 0.35,
        "net_arch": [256, 128, 64],
        "activation": "tanh",
        "log_std_init": -1.3,
        "learning_rate": 2e-4,
        "preferred_speed_mps": 0.35,
    }
    with open(os.path.join(artifacts_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    vec_env = SubprocVecEnv([make_env(i) for i in range(args.n_envs)])

    policy_kwargs = dict(
        activation_fn=nn.Tanh,
        log_std_init=-1.3,
        net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64]),
    )
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=2e-4,
        n_steps=2048,
        batch_size=1024,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.002,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=device,
    )

    pretrain_actor(model, device=device, epochs=35)
    model.save(os.path.join(artifacts_dir, "pretrain_base.zip"))

    logger = configure(artifacts_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(logger)

    eval_freq_calls = max(1, 30_000 // args.n_envs)
    eval_cb = DistanceEvalCallback(eval_freq=eval_freq_calls, artifacts_dir=artifacts_dir)

    print("\n=== ЭТАП 2: PPO RL-дообучение в точном MuJoCo ===")
    t0 = time.time()
    try:
        model.learn(total_timesteps=args.steps, callback=eval_cb, progress_bar=False)
    except KeyboardInterrupt:
        print("\nОбучение остановлено пользователем (Ctrl+C), сохраняю текущий прогресс...")
    finally:
        elapsed = time.time() - t0
        print(f"\nВремя этапа 2: {elapsed / 60.0:.1f} мин")
        final_zip = os.path.join(artifacts_dir, "final_model.zip")
        final_npz = os.path.join(artifacts_dir, "policy_weights.npz")
        model.save(final_zip)
        export_policy_npz(model, final_npz)
        save_plots_and_summary(artifacts_dir, eval_cb)
        try:
            vec_env.close()
        except Exception:
            pass
        print(f"Все артефакты и графики сохранены в: {artifacts_dir}")


if __name__ == "__main__":
    main()

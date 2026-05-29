"""Training script for QuadXDomeWaypointsEnv using SAC.

Usage:
    python src/train_dome_waypoints.py
    python src/train_dome_waypoints.py --config configs/dome_waypoints.yaml
    python src/train_dome_waypoints.py --total_timesteps 500000 --n_envs 4
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import wandb
import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    EvalCallback,
)
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from wandb.integration.sb3 import WandbCallback

import gymnasium as gym
import PyFlyt.gym_envs  # noqa: F401 — registers PyFlyt/QuadX-DomeWaypoints-v1
from PyFlyt.gym_envs.quadx_envs.quadx_dome_waypoints_env import QuadXDomeWaypointsEnv

ENV_ID = "PyFlyt/QuadX-DomeWaypoints-v1"
DEFAULT_CONFIG = Path(__file__).parent.parent / "configs" / "dome_waypoints.yaml"


# ── Callbacks ─────────────────────────────────────────────────────────────────

class RewardLoggingCallback(BaseCallback):
    """Log reward_components/* and num_waypoints_collected to TensorBoard/WandB."""

    def _on_step(self) -> bool:
        infos = self.locals["infos"]

        if "reward_components" in infos[0]:
            for key in infos[0]["reward_components"]:
                avg = np.mean([
                    info["reward_components"][key]
                    for info in infos
                    if "reward_components" in info and key in info["reward_components"]
                ])
                self.logger.record(f"reward_components/{key}", avg)

        if "num_waypoints_collected" in infos[0]:
            avg_collected = np.mean([
                info.get("num_waypoints_collected", 0) for info in infos
            ])
            self.logger.record("env/num_waypoints_collected", avg_collected)

        return True


class MyWandbCallback(WandbCallback):
    def _on_step(self) -> bool:
        if self.num_timesteps % 5000 == 0:
            rewards = self.locals.get("rewards")
            print(f"[WandB] step={self.num_timesteps}  last_reward={rewards}")
        return super()._on_step()


class VideoCheckpointCallback(BaseCallback):
    """Save a model checkpoint and record a video episode at regular intervals.

    After each checkpoint, runs one episode with the current policy in a
    separate headless env, writes an MP4, and logs it to WandB.
    """

    def __init__(
        self,
        save_freq: int,
        save_path: str,
        env_params: dict,
        video_dir: str,
        fps: int = 30,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.env_params = env_params
        self.video_dir = video_dir
        self.fps = fps

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            ckpt_path = os.path.join(
                self.save_path, f"model_{self.num_timesteps}_steps"
            )
            self.model.save(ckpt_path)
            if self.verbose:
                print(f"[Checkpoint] Saved {ckpt_path}.zip at step {self.num_timesteps}")

            self._record_video()

        return True

    def _record_video(self) -> None:
        """Run one episode with the current policy and save an MP4."""
        try:
            import imageio.v2 as imageio
        except ImportError:
            print("[VideoCheckpoint] imageio not installed — skipping video. pip install imageio[ffmpeg]")
            return

        os.makedirs(self.video_dir, exist_ok=True)

        # Headless env — capture_frame() uses PyBullet's software renderer
        rec_env = QuadXDomeWaypointsEnv(**self.env_params)
        obs, _ = rec_env.reset(seed=self.num_timesteps)

        frames = [rec_env.capture_frame()]
        total_reward = 0.0

        while True:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, term, trunc, _ = rec_env.step(action)
            frames.append(rec_env.capture_frame())
            total_reward += float(reward)
            if term or trunc:
                break

        rec_env.close()

        video_path = os.path.join(
            self.video_dir, f"step_{self.num_timesteps:08d}.mp4"
        )
        writer = imageio.get_writer(video_path, fps=self.fps, codec="libx264", quality=8)
        for frame in frames:
            writer.append_data(frame)
        writer.close()

        print(
            f"[VideoCheckpoint] Saved {video_path} "
            f"({len(frames)} frames, reward={total_reward:.1f})"
        )

        # Log to WandB if a run is active
        if wandb.run is not None:
            wandb.log(
                {
                    "video/episode": wandb.Video(video_path, fps=self.fps, format="mp4"),
                    "video/total_reward": total_reward,
                    "video/num_frames": len(frames),
                },
                step=self.num_timesteps,
            )


# ── Training ──────────────────────────────────────────────────────────────────

def train(config: dict) -> None:
    """Run single-agent SAC training for QuadXDomeWaypointsEnv."""
    total_timesteps: int = config.get("total_timesteps", 1_000_000)
    n_envs: int = config.get("n_envs", 8)
    output_folder: str = config.get("output_folder", "results/dome_waypoints")
    video_freq: int = config.get("video_freq", 50_000)
    sac_hyperparams: dict = config.get("sac_hyperparams") or {}
    env_params: dict = config.get("env_params") or {}

    timestamp = datetime.now().strftime("%m.%d.%Y_%H.%M")
    save_dir = os.path.join(output_folder, f"save-dome_waypoints-{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    # Save the effective config alongside the run
    with open(os.path.join(save_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    print(f"[INFO] Training QuadXDomeWaypointsEnv for {total_timesteps:,} steps")
    print(f"[INFO] Saving to {save_dir}")

    # ── Environments ──────────────────────────────────────────────────────────
    def make_env(seed: int = 0):
        def _init():
            env = gym.make(ENV_ID, **env_params)
            env.reset(seed=seed)
            return env
        return _init

    train_env = VecMonitor(DummyVecEnv([make_env(seed=i) for i in range(n_envs)]))
    eval_env = VecMonitor(DummyVecEnv([make_env(seed=9999)]))

    # ── WandB ─────────────────────────────────────────────────────────────────
    wb = wandb.init(
        project="dome-waypoints",
        name=f"sac-dome_waypoints-{timestamp}",
        sync_tensorboard=True,
        config={
            "algo": "SAC",
            "env": ENV_ID,
            "total_timesteps": total_timesteps,
            "n_envs": n_envs,
            **env_params,
            **sac_hyperparams,
        },
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = SAC(
        policy="MlpPolicy",
        env=train_env,
        verbose=1,
        tensorboard_log=os.path.join(save_dir, "tb_logs"),
        **sac_hyperparams,
    )

    checkpoint_freq = max(video_freq // n_envs, 1)

    # ── Callbacks ─────────────────────────────────────────────────────────────
    callbacks = CallbackList([
        EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(save_dir, "best_model"),
            log_path=os.path.join(save_dir, "eval"),
            eval_freq=max(10_000 // n_envs, 1),
            n_eval_episodes=5,
            deterministic=True,
            render=False,
        ),
        VideoCheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=os.path.join(save_dir, "checkpoints"),
            env_params=env_params,
            video_dir=os.path.join(save_dir, "videos"),
            fps=30,
            verbose=1,
        ),
        RewardLoggingCallback(),
        MyWandbCallback(verbose=0),
    ])

    # ── Train ─────────────────────────────────────────────────────────────────
    model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=False)

    # ── Save final model ──────────────────────────────────────────────────────
    final_path = os.path.join(save_dir, "final_model")
    model.save(final_path)
    wb.save(final_path + ".zip")
    print(f"[INFO] Final model saved to {final_path}.zip")

    wandb.finish()
    train_env.close()
    eval_env.close()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAC training for QuadX Dome Waypoints")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help=f"Path to YAML config (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument("--total_timesteps", type=int)
    parser.add_argument("--n_envs", type=int)
    parser.add_argument("--output_folder", type=str)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    for key in ("total_timesteps", "n_envs", "output_folder"):
        val = getattr(args, key)
        if val is not None:
            cfg[key] = val

    print(f"[INFO] Loaded config from {args.config}")
    train(cfg)

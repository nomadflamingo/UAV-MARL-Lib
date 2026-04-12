"""Unified multi-agent evaluation script.

Loads trained models from a save directory, runs statistical evaluation,
and optionally records video. Works with any PettingZoo env in ENV_REGISTRY.

Usage:
    # Statistical eval (headless)
    python eval.py --save_dir results/pe/save-pursuit_evasion-04.01.2026_22.29

    # With video recording
    python eval.py --save_dir results/pe/save-pursuit_evasion-04.01.2026_22.29 --record

    # Visual (GUI) mode
    python eval.py --save_dir results/pe/save-pursuit_evasion-04.01.2026_22.29 --visual

    # Override env type and model template
    python eval.py --save_dir ./my_models --env pursuit_evasion --num_episodes 50
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter

import numpy as np
from stable_baselines3 import SAC

from PyFlyt.pz_envs import MAFixedwingDogfightEnvV2
from PyFlyt.pz_envs.quadx_envs.ma_combat_env import CombatWaypointPursuitEnv
from PyFlyt.pz_envs.quadx_envs.ma_quadx_hover_env import MAQuadXHoverEnv
from PyFlyt.pz_envs.quadx_envs.ma_quadx_dogfight_env import MAQuadXDogfightEnv
from PyFlyt.pz_envs.quadx_envs.ma_quadx_pursuit_evasion_env import MAQuadXPursuitEvasionEnv

ENV_REGISTRY = {
    "dogfight_FW": MAFixedwingDogfightEnvV2,
    "dogfight": MAFixedwingDogfightEnvV2,
    "dogfight_QX": MAQuadXDogfightEnv,
    "combat": CombatWaypointPursuitEnv,
    "hover": MAQuadXHoverEnv,
    "pursuit_evasion": MAQuadXPursuitEvasionEnv,
}


def extract_env_type(path: str) -> str:
    """Extract env type from a save directory name like 'save-pursuit_evasion-...'."""
    match = re.search(r"save-([^-]+(?:_[^-]+)*)-", path)
    if not match:
        raise ValueError(f"Could not extract environment type from path: {path}")
    env_type = match.group(1)
    if env_type not in ENV_REGISTRY:
        raise KeyError(
            f"Environment type '{env_type}' not found in ENV_REGISTRY. "
            f"Available: {list(ENV_REGISTRY.keys())}"
        )
    return env_type


def _find_latest_br(save_dir: str, agent_id: int) -> str | None:
    """Find the latest best-response model file for an agent (agent_X_br_YYYY.zip)."""
    import glob
    pattern = os.path.join(save_dir, f"agent_{agent_id}_br_*.zip")
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None


def load_models(save_dir: str, agent_names: list[str], template: str) -> dict[str, SAC]:
    """Load SAC models for each agent from save_dir.

    Tries the template first (e.g. final_agent_0_model.zip).
    Falls back to the latest best-response checkpoint (agent_0_br_YYYY.zip).
    """
    models = {}
    for i, agent in enumerate(agent_names):
        model_path = os.path.join(save_dir, template.format(agent_num=i))
        if not os.path.exists(model_path):
            # Fall back to latest best-response model
            model_path = _find_latest_br(save_dir, i)
            if model_path is None:
                raise FileNotFoundError(
                    f"No model found for agent {i} in {save_dir} "
                    f"(tried template '{template}' and agent_{i}_br_*.zip)"
                )
        models[agent] = SAC.load(model_path)
        print(f"[INFO] Loaded model for {agent}: {model_path}", flush=True)
    return models


def evaluate(env, models: dict, num_episodes: int = 100, seed: int | None = None):
    """Run statistical evaluation.

    If the env implements interpret_outcome(), uses it to classify results.
    Otherwise falls back to counting which agent got the highest reward.

    Returns:
        list[dict]: Per-episode result dicts with keys 'outcome', 'rewards', 'steps'.
    """
    has_interpret = hasattr(env, "interpret_outcome")
    results = []

    for ep in range(num_episodes):
        ep_seed = (seed + ep) if seed is not None else None
        obs, _ = env.reset(seed=ep_seed)

        all_agents = list(env.agents)
        cumulative_rewards = {agent: 0.0 for agent in all_agents}
        infos = {agent: {} for agent in all_agents}
        steps = 0

        while env.agents:
            actions = {
                agent: models[agent].predict(obs[agent], deterministic=True)[0]
                for agent in env.agents
            }
            obs, rewards, terms, truncs, step_infos = env.step(actions)
            for agent in rewards:
                cumulative_rewards[agent] += rewards[agent]
            infos.update(step_infos)
            steps += 1

        if has_interpret:
            outcome = env.interpret_outcome(infos)
        else:
            outcome = "unknown"

        result = {
            "episode": ep + 1,
            "outcome": outcome,
            "rewards": {agent: round(r, 2) for agent, r in cumulative_rewards.items()},
            "steps": steps,
        }
        results.append(result)
        print(f"  Episode {ep + 1:3d}/{num_episodes} — {outcome} ({steps} steps)", flush=True)

    return results


def record_video(env, models: dict, output_dir: str, num_episodes: int = 1,
                 fps: int = 30, seed: int | None = None):
    """Record evaluation episodes to MP4 video files.

    Requires the env to implement capture_frame() -> np.ndarray.
    """
    import imageio.v2 as imageio

    if not hasattr(env, "capture_frame"):
        print("[WARN] Environment does not implement capture_frame(), skipping video.")
        return

    os.makedirs(output_dir, exist_ok=True)

    for ep in range(num_episodes):
        ep_seed = (seed + ep) if seed is not None else None
        obs, _ = env.reset(seed=ep_seed)

        frames = [env.capture_frame()]

        while env.agents:
            actions = {
                agent: models[agent].predict(obs[agent], deterministic=True)[0]
                for agent in env.agents
            }
            obs, rewards, terms, truncs, infos = env.step(actions)
            frames.append(env.capture_frame())

        path = os.path.join(output_dir, f"episode_{ep + 1:02d}.mp4")
        writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8)
        for f in frames:
            writer.append_data(f)
        writer.close()
        print(f"[INFO] Saved video: {path} ({len(frames)} frames)")


def run_visual(env, models: dict, seed: int | None = 7):
    """Run a single episode with the pybullet GUI for live viewing."""
    obs, _ = env.reset(seed=seed)
    infos = {}

    while env.agents:
        actions = {
            agent: models[agent].predict(obs[agent], deterministic=True)[0]
            for agent in env.agents
        }
        obs, rewards, terms, truncs, infos = env.step(actions)
        time.sleep(1.0 / 40)

    if hasattr(env, "interpret_outcome"):
        print(f"[INFO] Outcome: {env.interpret_outcome(infos)}")
    if hasattr(env, "render_trajectory"):
        env.render_trajectory()


def print_summary(results: list[dict]):
    """Print a summary table of evaluation results."""
    outcomes = Counter(r["outcome"] for r in results)
    total = len(results)

    print(f"\n{'='*50}")
    print(f"  Evaluation Summary ({total} episodes)")
    print(f"{'='*50}")
    for outcome, count in sorted(outcomes.items()):
        pct = 100.0 * count / total
        print(f"  {outcome:20s}: {count:4d}  ({pct:5.1f}%)")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-agent evaluation")
    parser.add_argument("--save_dir", required=True, help="Path to training save directory")
    parser.add_argument("--env", default=None, help="Override env type (auto-detected from save_dir if omitted)")
    parser.add_argument("--model_template", default="final_agent_{agent_num}_model.zip",
                        help="Filename template for agent models")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of eval episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_duration", type=float, default=30.0,
                        help="Max episode duration in seconds")

    parser.add_argument("--record", action="store_true", help="Record episodes to video")
    parser.add_argument("--record_episodes", type=int, default=3,
                        help="Number of episodes to record (with --record)")
    parser.add_argument("--video_dir", default=None, help="Video output directory (default: save_dir/videos)")
    parser.add_argument("--fps", type=int, default=30, help="Video FPS")

    parser.add_argument("--visual", action="store_true", help="Run one episode with GUI for live viewing")

    args = parser.parse_args()

    # Resolve env type
    env_name = args.env or extract_env_type(args.save_dir)
    env_class = ENV_REGISTRY[env_name]
    print(f"[INFO] Environment: {env_name}")

    # Create env and load models
    render_mode = "human" if args.visual else None
    env = env_class(render_mode=render_mode, max_duration_seconds=args.max_duration)
    env.reset(seed=args.seed)
    models = load_models(args.save_dir, list(env.agents), args.model_template)

    if args.visual:
        run_visual(env, models, seed=args.seed)
    else:
        # Statistical evaluation
        results = evaluate(env, models, num_episodes=args.num_episodes, seed=args.seed)
        print_summary(results)

        # Save raw results
        results_path = os.path.join(args.save_dir, "eval_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[INFO] Raw results saved to {results_path}")

        # Video recording
        if args.record:
            video_dir = args.video_dir or os.path.join(args.save_dir, "videos")
            record_env = env_class(render_mode="human", max_duration_seconds=args.max_duration)
            record_video(record_env, models, video_dir,
                         num_episodes=args.record_episodes, fps=args.fps, seed=args.seed)
            record_env.close()

    env.close()
    print("[INFO] Done.")

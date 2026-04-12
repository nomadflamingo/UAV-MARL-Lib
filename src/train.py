"""Launch MARL training from a YAML config file.

Usage:
    python train.py                               # uses configs/pursuit_evasion.yaml
    python train.py --config configs/my_run.yaml
    python train.py --config configs/pursuit_evasion.yaml --total_timesteps 50000 --strategy fp
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from training.train_ma_envs import train

DEFAULT_CONFIG = Path(__file__).parent.parent / "configs" / "pursuit_evasion.yaml"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MARL training entry point")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG),
                        help="Path to YAML training config (default: configs/pursuit_evasion.yaml)")
    parser.add_argument("--env", help="Environment key (overrides config)")
    parser.add_argument("--strategy", help="Training strategy (overrides config)")
    parser.add_argument("--total_timesteps", type=int, help="Total env steps (overrides config)")
    parser.add_argument("--update_interval", type=int, help="Steps per FSP iteration (overrides config)")
    parser.add_argument("--n_envs", type=int, help="Number of parallel envs (overrides config)")
    parser.add_argument("--output_folder", help="Output directory (overrides config)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Apply CLI overrides (only if explicitly provided)
    for key in ("env", "strategy", "total_timesteps", "update_interval", "n_envs", "output_folder"):
        val = getattr(args, key)
        if val is not None:
            cfg[key] = val

    print(f"[INFO] Loaded config from {args.config}")
    train(cfg)

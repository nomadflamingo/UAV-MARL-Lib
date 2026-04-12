"""Launch MARL training from a YAML config file.

Usage:
    python train.py                               # uses configs/pursuit_evasion.yaml
    python train.py --config configs/my_run.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from test_ma_envs import train

DEFAULT_CONFIG = Path(__file__).parent / "configs" / "pursuit_evasion.yaml"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MARL training entry point")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Path to YAML training config (default: configs/pursuit_evasion.yaml)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print(f"[INFO] Loaded config from {args.config}")
    train(**cfg)

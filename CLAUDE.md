# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

UAV-MARL-Lib is a multi-agent reinforcement learning library for UAV (drone) environments, built on top of the PyFlyt flight simulator. It extends PyFlyt with additional multi-agent PettingZoo environments and game-theoretic training strategies (fictitious play, self-play variants) using Stable-Baselines3 SAC.

## Build and Install

```bash
pip install -e .           # Editable install (core dependencies)
pip install -e ".[dev]"    # With dev tools (pytest, matplotlib, wandb, pre-commit)
```

System dependencies for PyBullet GUI: see `Dockerfile` for required OpenGL/mesa packages on Linux.

## Running Tests

```bash
# All tests 
pytest tests/*.py -vvv

# Single test file
pytest tests/test_pz_envs.py -vvv
pytest tests/test_gym_envs.py -vvv

# Single test
pytest tests/test_pz_envs.py::test_parallel_api -vvv
```

**After making any code changes, always run the relevant test file before finishing.** For changes to the pursuit-evasion environment, run `pytest tests/test_pursuit_evasion.py -vvv`.

## Linting

Pre-commit hooks enforce: black, isort (profile=black), flake8 (max-line-length=456), pydocstyle (google), pyupgrade (py38+), pyright, codespell.

```bash
pre-commit run --all-files
```

## Training

```bash
# Main multi-agent training entry point
python test_ma_envs.py --env dogfight_FW --total_timesteps 5000 --update_interval 1000

# Single-agent training
python test_sa_envs.py
```

Key CLI args for `test_ma_envs.py`: `--env`, `--strategy`, `--num_agents`, `--total_timesteps`, `--update_interval`, `--n_envs`, `--output_folder`, `--flight_mode`.

## Evaluation

```bash
python run_ma_policy.py   # Loads saved models, runs competitive eval, plots results
```

Edit `save_dir` and `model_filename_template` in the script to point at your trained models.

## Architecture

### Two-layer environment hierarchy

**Physics layer** (`PyFlyt/core/`): PyBullet-based flight simulation.
- `Aviary` — central physics manager (240Hz physics, configurable agent Hz)
- Drone implementations: `QuadX`, `Fixedwing`, `Rocket` in `core/drones/`
- Vehicle parameters defined in YAML configs under `models/vehicles/`

**Environment layer** — Gymnasium and PettingZoo wrappers over the physics:
- **Single-agent** (`PyFlyt/gym_envs/`): Standard Gymnasium envs (hover, waypoints, gates, pole balance, ball-in-cup). Base class: `QuadXBaseEnv`.
- **Multi-agent** (`PyFlyt/pz_envs/`): PettingZoo `ParallelEnv` implementations. Key envs:
  - `MAFixedwingDogfightEnv` (V2) — team-based aerial dogfight with configurable team size, damage model, lethal distance/angle
  - `CombatWaypointPursuitEnv` — ego pursues waypoints while adversary pursues ego (Dict observation space)
  - `MAQuadXHoverEnv` — multi-agent cooperative hovering

### Game-theoretic training system (`PyFlyt/marl_wrappers/selfplay.py`)

This is the core MARL contribution. The wrapper chain:

```
FictitiousPlayEnv (training orchestrator)
  └─ VecMonitor(DummyVecEnv([
      └─ MASelfPlayEnv (wraps multi-agent env for single-agent SB3 training)
        └─ PettingZoo ParallelEnv
    ]))
```

- `SelfPlayEnv` / `MASelfPlayEnv` — adapts a PettingZoo env to Gymnasium for SB3, sampling opponent actions from fixed policies
- `FictitiousPlayEnv` — orchestrates iterative training: maintains policy archives per agent, updates policy distributions, creates averaged opponent policies

**Strategy registry** (prefix `s` = self-play variant):
- `vp` / `svp` — Vanilla Play (only latest policy)
- `fp` / `sfp` — Fictitious Play (cumulative frequency matching)
- `dp` / `sdp` — Delta-Uniform Play (uniform over last k=10 policies)

Self-play strategies (`s*`) train against the agent's *own* past policies; non-self-play strategies train against the *opponent's* past policies.

### Registries

Environments and strategies are selected via dicts in `test_ma_envs.py`:
- `ENV_REGISTRY` — maps string keys to environment classes
- `STRAT_REGISTRY` — list of valid strategy codes

### Logging

WandB + TensorBoard. Custom callbacks in `test_ma_envs.py`: `RewardLoggingCallback` (logs `reward_components/*` from info dict), `MyWandbCallback`.

### Key env design details

- Observation space: attitude (12D Euler / 13D quaternion) + auxiliary state + health (for competitive envs)
- Action space: 4D for QuadX (angular rates + thrust), 4-6D for Fixedwing
- `combat` env uses `MultiInputPolicy` (Dict obs); all others use `MlpPolicy`
- Competitive envs signal game outcome via `team_win` flag in the info dict

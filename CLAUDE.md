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
# Multi-agent training (config-driven)
python src/train.py --config configs/pursuit_evasion.yaml

# Override config values via CLI
python src/train.py --config configs/pursuit_evasion.yaml --total_timesteps 50000 --strategy fp
```

Training configs are YAML files in `configs/`. Key config keys: `env`, `strategy`, `total_timesteps`, `update_interval`, `n_envs`, `output_folder`, `policy`, `sac_hyperparams`, `env_params`.

The core training logic lives in `src/training/train_ma_envs.py`.

## Evaluation

```bash
# Statistical eval (headless, saves eval_results.json to save_dir)
python src/eval.py --save_dir results/pe/save-pursuit_evasion-04.01.2026_22.29

# Record video (requires env to implement capture_frame())
python src/eval.py --save_dir ... --record --record_episodes 3

# Live GUI viewing
python src/eval.py --save_dir ... --visual
```

Key CLI args: `--save_dir` (required), `--env` (auto-detected from dir name), `--model_template`, `--num_episodes` (default 10), `--seed`, `--max_duration`, `--record`, `--visual`.

Model loading: tries the template first (`final_agent_{agent_num}_model.zip`), then falls back to the latest `agent_X_br_*.zip` best-response checkpoint.

## Plotting

```bash
# Plot eval reward curves
python src/plotting/plot_rewards.py --run_dir results/pe/save-pursuit_evasion-...

# Plot win-rate comparison across strategies
python src/plotting/plot_win_rates.py --results run1/eval_results.json run2/eval_results.json --labels "VP" "FP"
```

## Project Structure

```
├── PyFlyt/                          # Library source
│   ├── core/                        # Physics layer (Aviary, drones)
│   ├── gym_envs/                    # Single-agent Gymnasium envs
│   ├── pz_envs/                     # Multi-agent PettingZoo envs
│   └── marl_wrappers/               # Self-play / fictitious play wrappers
├── src/
│   ├── train.py                     # Training entry point (loads YAML config)
│   ├── eval.py                      # Evaluation entry point (stats, video, visual)
│   ├── training/
│   │   └── train_ma_envs.py         # Core MA training logic, ENV_REGISTRY, callbacks
│   ├── plotting/
│   │   ├── plot_rewards.py          # Eval reward curves
│   │   └── plot_win_rates.py        # Win-rate bar charts from eval_results.json
│   └── visualization/
│       └── view_env.py              # Quick visual test with random actions
├── configs/                         # YAML training configs
├── tests/                           # pytest test suite
└── results/                         # Training outputs (gitignored)
```

## Architecture

### Two-layer environment hierarchy

**Physics layer** (`PyFlyt/core/`): PyBullet-based flight simulation.
- `Aviary` — central physics manager, inherits from `BulletClient` (240Hz physics, configurable agent Hz)
- Drone implementations: `QuadX`, `Fixedwing`, `Rocket` in `core/drones/`

**Environment layer** — Gymnasium and PettingZoo wrappers over the physics:
- **Single-agent** (`PyFlyt/gym_envs/`): Standard Gymnasium envs (hover, waypoints). Base class: `QuadXBaseEnv`.
- **Multi-agent** (`PyFlyt/pz_envs/`): PettingZoo `ParallelEnv` implementations. Key envs:
  - `MAFixedwingDogfightEnvV2` — team-based aerial dogfight with configurable team size, damage model, lethal distance/angle
  - `MAQuadXPursuitEvasionEnv` — configurable NvN pursuit-evasion in a spherical arena (primary active env)
  - `CombatWaypointPursuitEnv` — ego pursues waypoints while adversary pursues ego (Dict observation space)
  - `MAQuadXHoverEnv` — multi-agent cooperative hovering

Competitive envs implement `capture_frame()` (returns RGB array for video recording) and `interpret_outcome(infos)` (returns outcome string like `"pursuers_win"`, `"evaders_win"`, `"draw"`). These are used by `src/eval.py`.

### PettingZoo agent lifecycle (important)

**PettingZoo `ParallelEnv` removes terminated/truncated agents from `env.agents` after each `step()`.** The correct episode loop pattern is:

```python
obs, _ = env.reset()
while env.agents:          # NOT while not all(dones.values())
    actions = {agent: ... for agent in env.agents}
    obs, rewards, terms, truncs, infos = env.step(actions)
```

Do **not** track a separate `dones` dict — agents disappear from `env.agents` on termination, so a `dones`-based loop will spin forever once agents are removed but `dones` still has `False` entries for them.

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

Environments and strategies are selected via dicts in `src/training/train_ma_envs.py`:
- `ENV_REGISTRY` — maps string keys to environment classes
- Strategy codes validated in training logic

### Logging

WandB + TensorBoard. Custom callbacks in `src/training/train_ma_envs.py`: `RewardLoggingCallback` (logs `reward_components/*` from info dict), `MyWandbCallback`.

### Key env design details

- Observation space: attitude (12D Euler / 13D quaternion) + auxiliary state + health (for competitive envs)
- Action space: 4D for QuadX (angular rates + thrust), 4-6D for Fixedwing
- `combat` env uses `MultiInputPolicy` (Dict obs); all others use `MlpPolicy`
- Pursuit-evasion env loads defaults from `configs/pursuit_evasion.yaml`; constructor kwargs override any config value

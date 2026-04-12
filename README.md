# UAV Multi-Agent Reinforcement Learning Library

A multi-agent reinforcement learning library for UAV (drone) environments, built on top of the [PyFlyt](https://github.com/jjshoots/PyFlyt) flight simulator. It extends PyFlyt with additional multi-agent PettingZoo environments and game-theoretic training strategies (fictitious play, self-play variants) using Stable-Baselines3 SAC.

<p align="center">
    <img src="https://github.com/jjshoots/PyFlyt/blob/master/readme_assets/pyflyt_cover_photo.png?raw=true" width="600"/>
</p>

## How this relates to PyFlyt

[PyFlyt](https://github.com/jjshoots/PyFlyt) is a UAV flight simulator — it provides drone physics (via PyBullet) and environment interfaces for reinforcement learning, but no training code. You get `reset()` and `step()`, and bring your own algorithms.

This repo builds on top of PyFlyt (it contains a modified fork in `PyFlyt/`) and adds everything needed to actually train competitive multi-agent policies: 
- new environments (pursuit-evasion, quadcopter dogfight, capture-the-flag)
- game-theoretic training strategies (fictitious play, self-play, delta-uniform play)
- config-driven SAC training via Stable-Baselines3
- evaluation/logging tooling (WandB, TensorBoard, video recording).

The key addition is the iterative best-response training loop in `PyFlyt/marl_wrappers/selfplay.py`: rather than training all agents simultaneously, each agent takes turns training against a weighted mixture of its opponent's (or its own) past policies. This pushes agents toward robust strategies instead of overfitting to a single opponent.

## Installation

### Prerequisites

- Python 3.11+
- System dependencies for PyBullet GUI (Linux):

```bash
sudo apt-get install libgl1-mesa-glx libgl1-mesa-dev libglu1-mesa-dev \
    libglew-dev libosmesa6-dev libglfw3 libglfw3-dev
```

### Install

```bash
git clone <repo-url> && cd UAV-MARL-Lib

# Core install
pip install -e .

# With dev tools (pytest, matplotlib, wandb, pre-commit)
pip install -e ".[dev]"
```

### WandB Setup

Training logs metrics to [Weights & Biases](https://wandb.ai/). Before your first training run:

```bash
# Login (creates ~/.netrc with your API key)
wandb login

# Or set the API key directly
export WANDB_API_KEY=your_key_here
```

To disable WandB logging (e.g. for quick local experiments):

```bash
export WANDB_MODE=disabled
```

WandB runs are created under the project `overnight-<env_name>-project` with TensorBoard sync enabled. All training hyperparameters are logged to the run config automatically.

## Quick Start: Pursuit-Evasion

The pursuit-evasion environment is the primary active environment. Two pursuers try to capture two evaders within a spherical arena.

### 1. Train

```bash
python src/train.py --config configs/pursuit_evasion.yaml
```

This will:
- Load the config from `configs/pursuit_evasion.yaml`
- Train 4 agents (2 pursuers, 2 evaders) using iterative best-response
- Save model checkpoints to `results/pe/` (configurable via `output_folder` in config)
- Log metrics to WandB and TensorBoard

### 2. Evaluate

```bash
# Statistical evaluation (10 episodes by default, headless)
python src/eval.py --save_dir results/pe/save-pursuit_evasion-04.01.2026_22.29

# Watch a single episode in the PyBullet GUI
python src/eval.py --save_dir results/pe/save-pursuit_evasion-04.01.2026_22.29 --visual

# Record video
python src/eval.py --save_dir results/pe/save-pursuit_evasion-04.01.2026_22.29 --record
```

### 3. Plot Results

```bash
# Plot training reward curves
python src/plotting/plot_rewards.py --run_dir results/pe/save-pursuit_evasion-04.01.2026_22.29

# Plot win rates (after running eval)
python src/plotting/plot_win_rates.py \
    --results results/pe/save-pursuit_evasion-04.01.2026_22.29/eval_results.json \
    --labels "Delta-Uniform Play"
```

## Training Configuration

Training is driven by YAML config files in `configs/`. The entry point is `src/train.py`:

```bash
python src/train.py --config configs/pursuit_evasion.yaml
```

### Config file reference

| Key | Required | Description |
|-----|----------|-------------|
| `env` | yes | Environment name (key in `ENV_REGISTRY`: `pursuit_evasion`, `dogfight_FW`, `hover`, `combat`) |
| `strategy` | no | Training strategy. Default: `sdp`. Options: `vp`, `fp`, `dp` (vanilla/fictitious/delta-uniform play), `svp`, `sfp`, `sdp` (self-play variants) |
| `total_timesteps` | no | Total env steps across all agents and iterations. Default: 5000 |
| `update_interval` | no | Steps per agent per training iteration. Default: 1000 |
| `n_envs` | no | Number of parallel envs for vectorized training. Default: 8 |
| `output_folder` | no | Directory for saving models and logs. Default: `junk` |
| `policy` | no | SB3 policy class. Default: `MlpPolicy`. Use `MultiInputPolicy` for Dict obs spaces |
| `sac_hyperparams` | no | Dict of SAC hyperparameters (`learning_rate`, `buffer_size`, `batch_size`, `gamma`, `tau`, `ent_coef`, etc.) |
| `env_params` | no | Dict of kwargs passed to the environment constructor (agent counts, rewards, arena size, etc.) |

### Example config (`configs/pursuit_evasion.yaml`)

```yaml
env: pursuit_evasion
policy: MlpPolicy
strategy: dp
total_timesteps: 2000000
update_interval: 25000
n_envs: 8
output_folder: results/pe

sac_hyperparams:
  learning_rate: 0.0003
  buffer_size: 100000
  batch_size: 256

env_params:
  num_pursuers: 2
  num_evaders: 2
  capture_distance: 0.2
  flight_dome_size: 3.0
  max_duration_seconds: 30.0
  # ... see configs/pursuit_evasion.yaml for all options
```

### Self-play strategies

| Code | Name | Description |
|------|------|-------------|
| `vp` | Vanilla Play | Train against the opponent's latest policy only |
| `fp` | Fictitious Play | Train against cumulative frequency-weighted mix of all opponent policies |
| `dp` | Delta-Uniform Play | Train against uniform mix of opponent's last k=10 policies |
| `svp` | Self Vanilla Play | Like `vp` but against own past policies |
| `sfp` | Self Fictitious Play | Like `fp` but against own past policies |
| `sdp` | Self Delta-Uniform Play | Like `dp` but against own past policies |

## Evaluation Reference

The evaluation script is `src/eval.py`. It auto-detects the environment type from the save directory name.

```bash
python src/eval.py --save_dir <path> [options]
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--save_dir` | (required) | Path to training save directory containing model checkpoints |
| `--env` | auto-detected | Override environment type (e.g. `pursuit_evasion`, `dogfight_FW`) |
| `--model_template` | `final_agent_{agent_num}_model.zip` | Filename template for models. Falls back to latest `agent_X_br_*.zip` if not found |
| `--num_episodes` | 10 | Number of evaluation episodes |
| `--seed` | 42 | Random seed |
| `--max_duration` | 30.0 | Max episode duration in seconds |
| `--visual` | off | Run one episode with PyBullet GUI for live viewing |
| `--record` | off | Record episodes to MP4 video |
| `--record_episodes` | 3 | Number of episodes to record |
| `--video_dir` | `<save_dir>/videos` | Output directory for recorded videos |
| `--fps` | 30 | Video framerate |

### Output

Statistical evaluation saves `eval_results.json` in the save directory. Each entry contains the episode outcome (`pursuers_win`, `evaders_win`, `draw`), per-agent cumulative rewards, and step count. A summary table is printed to stdout.

## Project Structure

```
UAV-MARL-Lib/
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
└── results/                         # Training outputs (not checked in)
```

## Creating Custom Multi-Agent Environments

All multi-agent environments are PettingZoo `ParallelEnv` subclasses that share a common base. To create a new one, subclass `MAQuadXBaseEnv` and implement four methods. The pursuit-evasion env (`PyFlyt/pz_envs/quadx_envs/ma_quadx_pursuit_evasion_env.py`) is a good reference.

### 1. Subclass `MAQuadXBaseEnv`

The base class (`PyFlyt/pz_envs/quadx_envs/ma_quadx_base_env.py`) handles the Aviary physics, agent naming (`uav_0`, `uav_1`, ...), action spaces, and the `step()` loop. You provide the game logic:

```python
from PyFlyt.pz_envs.quadx_envs.ma_quadx_base_env import MAQuadXBaseEnv

class MyCustomEnv(MAQuadXBaseEnv):
    metadata = {
        "render_modes": ["human"],
        "name": "my_custom_env",
        "is_parallelizable": True,
    }

    def __init__(self, render_mode=None, **kwargs):
        num_agents = 4
        start_pos = np.zeros((num_agents, 3))  # overwritten in reset
        start_orn = np.zeros((num_agents, 3))

        super().__init__(
            start_pos=start_pos,
            start_orn=start_orn,
            flight_mode=0,               # 0 = angular rate + thrust
            flight_dome_size=3.0,
            max_duration_seconds=30.0,
            angle_representation="euler",
            agent_hz=40,
            render_mode=render_mode,
        )

        # Define your observation space (must be a gymnasium.spaces.Box)
        self._observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_size,), dtype=np.float64
        )
```

### 2. Implement required methods

Four methods must be implemented:

**`observation_space(agent)`** — return the observation space (same for all agents in most cases):

```python
def observation_space(self, agent=None):
    return self._observation_space
```

**`reset(seed, options)`** — generate spawn positions, call `begin_reset` / `end_reset`, return initial observations:

```python
def reset(self, seed=None, options=None):
    self.start_pos, self.start_orn = self._generate_spawn_positions(seed)
    super().begin_reset(seed, options or {})
    super().end_reset(seed, options or {})
    # Reset any game state (e.g. capture flags, score counters)
    observations = {ag: self.compute_observation_by_id(self.agent_name_mapping[ag])
                    for ag in self.agents}
    infos = {ag: {} for ag in self.agents}
    return observations, infos
```

**`compute_observation_by_id(agent_id)`** — build a flat numpy array for one agent. Use `self.compute_attitude_by_id(agent_id)` to get `(ang_vel, ang_pos, lin_vel, lin_pos, quaternion)` and `self.aviary.aux_state(agent_id)` for auxiliary state. Add any game-specific features (relative positions to opponents, boundary distance, etc.):

```python
def compute_observation_by_id(self, agent_id):
    ang_vel, ang_pos, lin_vel, lin_pos, _ = self.compute_attitude_by_id(agent_id)
    aux_state = self.aviary.aux_state(agent_id)
    self_obs = np.concatenate([ang_vel, ang_pos, lin_vel, lin_pos, aux_state,
                               self.past_actions[agent_id]])
    # ... append relative teammate/opponent positions, game state, etc.
    return np.concatenate([self_obs, ...])
```

**`compute_term_trunc_reward_info_by_id(agent_id)`** — return `(terminated, truncated, reward, info)` for one agent. The base `step()` calls this every physics substep and accumulates rewards. Put reward components in `info["reward_components"]` for logging:

```python
def compute_term_trunc_reward_info_by_id(self, agent_id):
    term = False
    trunc = self.step_count > self.max_steps
    reward = 0.0
    info = {}
    rwd = {}

    # Check game-ending conditions (capture, collision, out-of-bounds)
    # Assign dense and sparse rewards
    # ...

    reward = sum(rwd.values())
    info["reward_components"] = rwd
    return term, trunc, reward, info
```

### 3. Optional: shared per-step computation

Override `update_states()` to compute shared state once per physics substep (called before per-agent reward/obs computation). This avoids redundant work:

```python
def update_states(self):
    self._update_pairwise_distances()
    self._check_captures()
```

### 4. Optional: evaluation support

To enable `src/eval.py` video recording and outcome statistics, implement two additional methods:

- **`capture_frame()`** — return an RGB numpy array from a debug camera (see pursuit-evasion env for a PyBullet camera setup example)
- **`interpret_outcome(infos)`** — return an outcome string (e.g. `"pursuers_win"`, `"evaders_win"`, `"draw"`) from the final step's info dicts

### 5. Register and configure

**Add to `PyFlyt/pz_envs/__init__.py`:**

```python
from .quadx_envs.my_custom_env import MyCustomEnv
```

**Add to `ENV_REGISTRY` in `src/training/train_ma_envs.py`:**

```python
ENV_REGISTRY = {
    ...
    "my_custom_env": MyCustomEnv,
}
```

**Create a YAML config** in `configs/my_custom_env.yaml`:

```yaml
env: my_custom_env
policy: MlpPolicy
strategy: dp
total_timesteps: 2000000
update_interval: 25000
n_envs: 8
output_folder: results/my_custom

env_params:
  # any kwargs passed to MyCustomEnv.__init__
```

Then train with:

```bash
python src/train.py --config configs/my_custom_env.yaml
```

## PyFlyt

This project builds on the PyFlyt UAV simulator. Full PyFlyt documentation: [jjshoots.github.io/PyFlyt](https://jjshoots.github.io/PyFlyt/documentation.html)

```
@article{tai2023pyflyt,
  title={PyFlyt--UAV Simulation Environments for Reinforcement Learning Research},
  author={Tai, Jun Jet and Wong, Jim and Innocente, Mauro and Horri, Nadjim and Brusey, James and Phang, Swee King},
  journal={arXiv preprint arXiv:2304.01305},
  year={2023}
}
```

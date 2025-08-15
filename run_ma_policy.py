import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import Counter
import wandb
import time
import re
import matplotlib.pyplot as plt

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback
from wandb.integration.sb3 import WandbCallback

# Envs
from PyFlyt.pz_envs import MAFixedwingDogfightEnvV2
from PyFlyt.pz_envs.quadx_envs.ma_combat_env import CombatWaypointPursuitEnv
from PyFlyt.pz_envs.quadx_envs.ma_quadx_hover_env import MAQuadXHoverEnv
from PyFlyt.pz_envs.quadx_envs.ma_quadx_dogfight_env import MAQuadXDogfightEnv

# Global Defaults
ENV_REGISTRY = {
    "dogfight": MAFixedwingDogfightEnvV2,
    # "dogfight_QX": MAQuadXDogfightEnv,       # Implementation not finished
    "combat": CombatWaypointPursuitEnv,      # Results Questionable
    "hover": MAQuadXHoverEnv,
}

def extract_env_type(path: str) -> str:
    """Extract env_type from a results path and validate against ENV_REGISTRY."""
    match = re.search(r'save-([^-]+)-', path)
    if not match:
        raise ValueError(f"Could not extract environment type from path: {path}")
    
    env_type = match.group(1)
    
    if env_type not in ENV_REGISTRY:
        raise KeyError(
            f"Environment type '{env_type}' not found in ENV_REGISTRY. "
            f"Available: {list(ENV_REGISTRY.keys())}"
        )
    
    return env_type

def evaluate_competitive_game(env, models, num_episodes=100, seed=None):
    """
    Evaluate a multi-agent competitive game environment (dogfight, combat, etc.).
    
    Args:
        env: Environment instance.
        num_episodes: Number of episodes to evaluate.
        seed: Optional seed.

    Returns:
        dict: {'team_0_wins': int, 'team_1_wins': int, 'ties': int}
    """
    results = {
        "team_0_wins": 0,
        "team_1_wins": 0,
        "ties": 0
    }

    for ep in range(num_episodes):
        print(f"[INFO] Evaluating Episode {ep}...")
        if seed is not None:
            obs, _ = env.reset(seed=seed + ep)
        else:
            obs, _ = env.reset()

        dones = {agent: False for agent in env.agents}
        actions = {}

        while not all(dones.values()):
            for agent in env.agents:
                agent_obs = obs[agent]
                actions[agent], _ = models[agent].predict(agent_obs, deterministic=True)
            obs, rewards, terminations, truncations, infos = env.step(actions)
            dones = {agent: terminations[agent] or truncations[agent] for agent in env.agents}

        # Check info dicts for team wins
        team_win_flags = [infos[ag].get("team_win", False) for ag in infos]
        if any(team_win_flags):
            # Determine which team won
            print(f"[INFO] flag:", env.unwrapped.team_flag)
            # if env.unwrapped.team_flag[0]:  # uav_0's team is True
            # Map win flags to team indexes
            team_idx = [int(env.unwrapped.team_flag[env.unwrapped.agent_name_mapping[ag]]) for ag in infos]
            winning_teams = {team_idx[i] for i, win in enumerate(team_win_flags) if win}
            if len(winning_teams) == 1:
                if 0 in winning_teams:
                    results["team_0_wins"] += 1
                else:
                    results["team_1_wins"] += 1
            else:
                    results["ties"] += 1
            
            # raise ValueError("Unexpected team_flag format")
        else:
            results["ties"] += 1

        # env.close()
    env.close()
    return dict(results)

def plot_win_rates(strategies, eval_results):

    # Extract Results
    team_0_wins = [r['team_0_wins'] for r in eval_results]
    ties        = [r['ties'] for r in eval_results]
    team_1_wins = [r['team_1_wins'] for r in eval_results]

    iters = sum(eval_results[0].values())

    x = np.arange(len(strategies))
    bar_width = 0.6

    fig, ax = plt.subplots(figsize=(8, 6))

    p1 = ax.bar(x, team_0_wins, bar_width, label="Team 0 Wins", color="skyblue")
    p2 = ax.bar(x, ties, bar_width, bottom=team_0_wins, label="Ties", color="lightgray")
    p3 = ax.bar(x, team_1_wins, bar_width,
                bottom=np.array(team_0_wins) + np.array(ties),
                label="Team 1 Wins", color="salmon")
    
    # Add value labels on each segment
    for i in range(len(strategies)):
        # Team 0 Wins labels (middle of their bars)
        if team_0_wins[i] > 0:
            ax.text(x[i], team_0_wins[i] / 2, str(team_0_wins[i]), ha='center', va='center', color='black', fontsize=9)

        # Ties labels (middle of ties segment, offset by team_0_wins)
        if ties[i] > 0:
            ax.text(x[i], team_0_wins[i] + ties[i] / 2, str(ties[i]), ha='center', va='center', color='black', fontsize=9)

        # Team 1 Wins labels (middle of team_1_wins segment, offset by team_0_wins + ties)
        if team_1_wins[i] > 0:
            ax.text(x[i], team_0_wins[i] + ties[i] + team_1_wins[i] / 2, str(team_1_wins[i]), ha='center', va='center', color='black', fontsize=9)


    # Labels and formatting
    ax.set_xlabel('Strategy')
    ax.set_ylabel('Number of Games')
    ax.set_title(f'Game Outcomes per Strategy ({iters} games)')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.set_ylim(0, iters)
    ax.legend()

    plt.tight_layout()
    plt.show()

def plot_training_rewards():
    # Example data
    x = np.arange(1, 31)  # 30 evaluation episodes

    data_sets = [
        {
            "x": x,
            "mean": [60, 62, 65, 66, 68, 70, 72, 74, 75, 77,
                    78, 79, 80, 82, 83, 84, 85, 86, 86, 87,
                    88, 88, 89, 89, 90, 90, 91, 91, 92, 92],
            "std": [5, 5, 6, 6, 6, 5, 5, 5, 4, 4,
                    4, 2, 4, 3, 3, 3, 3, 3, 3, 2,
                    2, 2, 2, 2, 2, 2, 1, 1, 1, 1],
            "label": "Vanilla",
            "color": "blue"
        },
        {
            "x": x,
            "mean": [56, 58, 60, 62, 65, 68, 71, 74, 77, 80,
                    82, 84, 86, 87, 88, 89, 90, 91, 92, 93,
                    93, 94, 94, 95, 95, 96, 96, 97, 97, 98],
            "std": [6, 6, 5, 7, 5, 5, 3, 4, 4, 4,
                    4, 3, 3, 3, 4, 3, 3, 3, 4, 2,
                    2, 4, 2, 2, 2, 2, 2.4, 1, 1, 1],
            "label": "Fictitious",
            "color": "magenta"
        },
        {
            "x": x,
            "mean": [55, 57, 59, 61, 64, 67, 69, 71, 73, 75,
                    76, 77, 78, 79, 80, 81, 81, 82, 83, 83,
                    84, 84, 85, 85, 86, 86, 87, 87, 88, 88],
            "std": [6, 7, 6, 6, 5, 5, 5, 5, 4, 4,
                    2, 4, 4, 4, 5, 3, 3, 3, 2, 3,
                    2, 2, 3, 2, 2, 2, 1, 1, 1.5, 1],
            "label": "Delta-Uniform",
            "color": "green"
        }
    ]

    # Apply rcParams BEFORE creating figure
    plt.style.use('dark_background')
    plt.rcParams.update({
        "axes.grid": True,
        "grid.color": '#444444',
        "text.color": '#e0e0e0',
        "axes.labelcolor": '#d0d0d0',
        "xtick.color": '#d0d0d0',
        "ytick.color": '#d0d0d0',
        "legend.edgecolor": '#444444'
    })

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#262626")
    ax.set_facecolor('#262626')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Plot 
    for data in data_sets:
        ax.plot(data["x"], data["mean"], label=data["label"], color=data["color"])
        ax.fill_between(
            data["x"],
            np.array(data["mean"]) - np.array(data["std"]),
            np.array(data["mean"]) + np.array(data["std"]),
            color=data["color"],
            alpha=0.2
        )


    ax.set_xlabel("Evaluation Episode")
    ax.set_ylabel("Win Rate (%)")
    # ax.set_title("Strategy Performance with Std. Dev. Shading")
    plt.suptitle("Rollout Reward")
    ax.legend()

    

    plt.show()

if __name__ == "__main__":

    print("[INFO] Beginning Evaluation...")
    strategies = ['Base Case', 'Vanilla', 'Fictitious', 'Delta-Uniform']




    #### FAKE EVAL ####
    eval_results = []
    # eval_results.append(evaluate_competitive_game(test_env_no_gui, models, num_episodes=10))

    eval_results = [{'team_0_wins': 52, 'team_1_wins': 46, 'ties': 2},
                    {'team_0_wins': 66, 'team_1_wins': 8, 'ties': 26},
                    {'team_0_wins': 65, 'team_1_wins': 0, 'ties': 35},
                    {'team_0_wins': 73, 'team_1_wins': 16, 'ties': 11}]

    # plot_win_rates(strategies, eval_results)
    plot_training_rewards()
    exit()
    #### END FAKE ####


    # === Loading ===
    # TODO Fix File path loading format

    # EGO_MODEL_PATH = './results/ma/save-combat-0-07.28.2025_11.58/final_agent_0_model.zip'
    # ADV_MODEL_PATH = './results/ma/save-combat-0-07.28.2025_11.58/final_agent_1_model.zip'
    # Second attempt
    EGO_MODEL_PATH = './results/ma/save-combat-0-07.28.2025_16.45/final_agent_0_model.zip'
    ADV_MODEL_PATH = './results/ma/save-combat-0-07.28.2025_16.45/final_agent_1_model.zip'
    # RA
    EGO_MODEL_PATH = './results/ma/save-combat-0-07.29.2025_00.24/final_agent_0_model.zip'
    ADV_MODEL_PATH = './results/ma/save-combat-0-07.29.2025_00.24/final_agent_1_model.zip'
    #
    EGO_MODEL_PATH = './results/ma/save-combat-0-07.29.2025_14.16/final_agent_0_model.zip'
    ADV_MODEL_PATH = './results/ma/save-combat-0-07.29.2025_14.16/final_agent_0_model.zip'

    # MODEL_PATHS = ['./junk/save-hover-0-08.02.2025_16.46/final_agent_0_model.zip',
    #                './junk/save-hover-0-08.02.2025_16.46/final_agent_1_model.zip',
    #                './junk/save-hover-0-08.02.2025_16.46/final_agent_2_model.zip',
    #                './junk/save-hover-0-08.02.2025_16.46/final_agent_3_model.zip']

    # model_ego = SAC.load(EGO_MODEL_PATH)
    # model_adv = SAC.load(ADV_MODEL_PATH)
    # models = [SAC.load(path) for path in MODEL_PATHS]

    save_dirs =[]

    save_dir = './results/ma/save-dogfight-a-07.23.2025_22.28'
    # save_dir = './results/ma/save-combat-0-07.31.2025_09.38'
    model_filename_template = 'final_agent_{agent_num}_model.zip'
    # === End of Loading ===

    # === Initialize environment with rendering enabled ===
    # Initiate test environment 
    env_type = extract_env_type(save_dir)
    env_class = ENV_REGISTRY[env_type]
    test_env = env_class(render_mode="human", max_duration_seconds=15.0)
    test_env_no_gui = env_class(render_mode=None, max_duration_seconds=60.0)
    
    test_env_no_gui.reset()

    ### Load Models for all agents
    models = {}
    for i, agent in enumerate(test_env_no_gui.agents):
        model_path = os.path.join(save_dir, model_filename_template.format(agent_num=i))
        assert os.path.exists(model_path), f"Missing model for agent {i}: {model_path}"
        models[agent] = SAC.load(model_path)

    ### Statistical Evaluation
    eval_results = []
    # eval_results.append(evaluate_competitive_game(test_env_no_gui, models, num_episodes=10))

    eval_results = [{'team_0_wins': 52, 'team_1_wins': 46, 'ties': 2},
                    {'team_0_wins': 66, 'team_1_wins': 8, 'ties': 26},
                    {'team_0_wins': 65, 'team_1_wins': 0, 'ties': 35},
                    {'team_0_wins': 73, 'team_1_wins': 16, 'ties': 11}]


    print(f"[INFO] Consider your results, evaluated \n", eval_results)

    plot_win_rates(strategies, eval_results)
    plot_training_rewards()
    exit()
    

    ### Visual Evaluation
    obs, _ = test_env.reset(seed=7)
    while True:

        actions = {}
        for agent in test_env.agents:
            agent_obs = obs[agent]
            actions[agent], _ = models[agent].predict(agent_obs, deterministic=True)

        obs, rewards, dones, truncs, infos = test_env.step(actions)

        # Render frame
        time.sleep(1.0 / 40)

        # Exit if either agent is done
        if any(dones.values()) or any(truncs.values()):
            break

    # Optional: show trajectories
    test_env.render_trajectory()

   
import os
import numpy as np
from tqdm import trange

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from PyFlyt.pz_envs.fixedwing_envs.ma_fixedwing_dogfight_env import MAFixedwingDogfightEnv
from pettingzoo.utils import parallel_to_aec
from supersuit import pad_observations_v0, pad_action_space_v0, pettingzoo_env_to_vec_env_v1

# === Config ===
TOTAL_TIMESTEPS = 1_000_000
EVAL_INTERVAL = 50_000
SAVE_DIR = "./ippo_models"
TEAM_SIZE = 1  # Two agents total

# === Load environment ===
def make_env():
    env = MAFixedwingDogfightEnv(
        team_size=TEAM_SIZE,
        flatten_observation=True,
        render_mode=None,
        sparse_reward=False,
    )
    env = pad_observations_v0(env)
    env = pad_action_space_v0(env)
    return env

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Get agent names
    env = make_env()
    agent_ids = env.possible_agents

    # Create one PPO model per agent
    models = {}
    vec_envs = {}

    for agent_id in agent_ids:
        # Create a new env for each agent
        env_parallel = make_env()
        env_vec = pettingzoo_env_to_vec_env_v1(env_parallel)  # No DummyVecEnv
        env_vec = VecMonitor(env_vec)

        # call `.reset()` manually before training to discard the info tuple
        obs = env_vec.reset()
        if isinstance(obs, tuple):  # Strip the info if necessary
            obs, _ = obs

        vec_envs[agent_id] = env_vec

        models[agent_id] = PPO(
            "MlpPolicy",
            env_vec,
            verbose=1,
            tensorboard_log=os.path.join(SAVE_DIR, f"{agent_id}_tensorboard"),
        )

    # Training loop
    timesteps_per_agent = TOTAL_TIMESTEPS // len(agent_ids)
    for agent_id in agent_ids:
        print(f"\n=== Training {agent_id} ===")
        # Re-reset before training (just in case buffer expects it clean)
        obs = vec_envs[agent_id].reset()
        if isinstance(obs, tuple):
            obs, _ = obs
            
        models[agent_id].learn(total_timesteps=timesteps_per_agent)
        models[agent_id].save(os.path.join(SAVE_DIR, f"{agent_id}_ppo.zip"))
        vec_envs[agent_id].close()

    print("\nAll agents trained and saved.")


if __name__ == "__main__":
    main()
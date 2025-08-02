import os
import wandb
import argparse
import numpy as np
from datetime import datetime
# SB3
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
# Gym
import gymnasium as gym
from gymnasium import spaces
# Envs
from PyFlyt.pz_envs import MAFixedwingDogfightEnvV2
from PyFlyt.pz_envs.quadx_envs.ma_combat_env import CombatWaypointPursuitEnv
from PyFlyt.pz_envs.quadx_envs.ma_quadx_hover_env import MAQuadXHoverEnv
from PyFlyt.pz_envs.quadx_envs.ma_quadx_dogfight_env import MAQuadXDogfightEnv
# pz
from pettingzoo.test import parallel_api_test
# SP
from PyFlyt.marl_wrappers.selfplay import SelfPlayEnv, MASelfPlayEnv


# Global Defaults
ENV_REGISTRY = {
    "dogfight_FW": MAFixedwingDogfightEnvV2,
    "dogfight_QX": MAQuadXDogfightEnv,       # Implementation not finished
    "combat": CombatWaypointPursuitEnv,      # Results Questionable
    "hover": MAQuadXHoverEnv,
}

DEFAULT_ENV = 'hover'
DEFAULT_RETRAIN = False
DEFAULT_TRAINED_FOLDER = 'name'
DEFAULT_FLIGHT_MODE = 0
# DEFAULT_OUTPUT_FOLDER = 'results/ma'
DEFAULT_OUTPUT_FOLDER = 'junk'
DEFAULT_NUM_AGENTS = 4
DEFAULT_TOTAL_TIMESTEPS = int(1e5)
DEFAULT_UPDATE_INTERVAL = int(1_000)
DEFAULT_NUM_ENVS = 8

class RewardLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        for info in infos:
            if "reward_components" in info:
                for key, value in info["reward_components"].items():
                    self.logger.record(f"reward_components/{key}", value)
        return True

class RandomPolicy:
    """A dummy policy that returns random actions from a given action_space."""
    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, obs, deterministic=True):
        # SB3 expects a tuple (action, state)
        return self.action_space.sample(), None
    
def make_env(ma_env, train_agent_id: int, seed: int, n_envs: int, flight_mode: int):
    def _init():
        # ma_env = CombatWaypointPursuitEnv(render_mode=None, flight_mode=flight_mode)
        ma_env.reset()
        # random_opp = RandomPolicy(ma_env.action_space(ma_env.agents[1 - train_agent_id]))
        opp_policies = {
            i: RandomPolicy(ma_env.action_space(i))
            for i in range(ma_env.num_possible_agents) if i != train_agent_id
        }
        env = MASelfPlayEnv(ma_env, train_agent_id, opp_policies)
        env.reset(seed=seed + train_agent_id)
        return env
    return _init

##### TRAINING FUNCTION #################################################################################
def train(env=DEFAULT_ENV, 
          retrain=DEFAULT_RETRAIN, 
          trained_folder=DEFAULT_TRAINED_FOLDER,
          flight_mode=DEFAULT_FLIGHT_MODE, 
          output_folder=DEFAULT_OUTPUT_FOLDER, 
          num_agents=DEFAULT_NUM_AGENTS,
          total_timesteps=DEFAULT_TOTAL_TIMESTEPS,
          update_interval=DEFAULT_UPDATE_INTERVAL,
          n_envs=DEFAULT_NUM_ENVS,
          strategy="double_oracle",):
    """
    Modular training routine for a choice of "ENV_REGISTRY" and "STRAT_REGISTRY".
    """
    print(f"\n\n[INFO] Beginning {'re' if retrain else ''}training agents in the \'{env}\' environment using a \'{strategy}\' method.")
    
    agent_ids = list(range(num_agents))
    agent_names = [f"uav_{i}" for i in range(num_agents)]

    #################################
    ### INITIATE THE ENVIRONMENTS ###
    #################################
    env_class = ENV_REGISTRY[env]
    ma_env = env_class(render_mode=None, flight_mode=flight_mode)
    if env == 'hover':
        policy = 'MlpPolicy'
        target_reward = 1600
    elif env == 'dogfight':
        policy = 'MultiInputPolicy'
    elif env == 'combat':
        policy = 'MultiInputPolicy'
    else:
        print("[ERROR] This environment is not currently suited to train the environment,", env)
        exit()

    # Create File 
    save_dir = os.path.join(output_folder, env)
    save_dir = os.path.join(output_folder, 'save-'+env+'-'+str(flight_mode)+'-'+datetime.now().strftime("%m.%d.%Y_%H.%M"))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir+'/')

    #################################
    ###    LOAD/INITIATE MODELS   ###
    #################################
    vec_envs = {}
    eval_envs = {}
    models = {}
    for agent_id in agent_ids:
        # Create vectorized training environments
        vec_envs[agent_id] = VecMonitor(
            DummyVecEnv([make_env(ma_env, agent_id, seed=42 + agent_id, n_envs=n_envs, flight_mode=flight_mode) for _ in range(n_envs)])
        )
        # Check env access
        print(f'[INFO] Agent {agent_id} action space:', vec_envs[agent_id].action_space)
        print(f'[INFO] Agent {agent_id} observation space:', vec_envs[agent_id].observation_space)

        # Create evaluation environments
        eval_envs[agent_id] = VecMonitor(
            DummyVecEnv([
                make_env(ma_env, agent_id, seed=1000 + agent_id, n_envs=1, flight_mode=flight_mode)
            ])
        )

        # Train model
        models[agent_id] = SAC(
            policy=policy,
            env=vec_envs[agent_id],
            verbose=1,
            tensorboard_log=os.path.join(save_dir, f"tb_agent_{agent_id}")
        )


    #################################
    ###         CALLBACKS         ###
    #################################
    callbacks = {}
    for agent_id in agent_ids:
        callbacks[agent_id] = [
            EvalCallback(
                eval_envs[agent_id],
                best_model_save_path=os.path.join(save_dir, f"eval_agent_{agent_id}"),
                log_path=os.path.join(save_dir, f"eval_agent_{agent_id}"),
                eval_freq=10_000,
                deterministic=True,
                render=False,
            ),
            CheckpointCallback(
                save_freq=250_000 // n_envs,
                save_path=os.path.join(save_dir, f"checkpoints/agent_{agent_id}"),
                name_prefix=f"agent_{agent_id}"
            ),
            RewardLoggingCallback()
        ]

    # Averaging buffer
    average_policies = {agent_id: [] for agent_id in agent_ids}

    #################################
    ###       TRAINING LOOP       ###
    #################################
    n_iters = total_timesteps // update_interval
    for it in range(1, n_iters + 1):
        for agent_id in agent_ids:
            print(f"[Iter {it}/{n_iters}] ▶ Training Agent {agent_id}")
            models[agent_id].learn(
                total_timesteps=update_interval,
                reset_num_timesteps=False,
                callback=callbacks[agent_id]
            )

            # Optionally, broadcast policy to all opponents
            if strategy == "double_oracle":
                for other_id in agent_ids:
                    if other_id != agent_id:
                        for env in vec_envs[other_id].envs:
                            env.opp_policy = models[agent_id]
            elif strategy == "fictitious_play":
                # Update average policy list
                average_policies[agent_id].append(models[agent_id])
                # Broadcast
                for other_id in agent_ids:
                    if other_id != agent_id:
                        def sample_average_policy(agent_list):
                            class AveragePolicy:
                                def __init__(self, policies):
                                    self.policies = policies
                                def predict(self, obs, deterministic=True):
                                    # Sample from historical best responses
                                    policy = np.random.choice(self.policies)
                                    return policy.predict(obs, deterministic)
                            return AveragePolicy(agent_list)

                        avg_policy = sample_average_policy(average_policies[agent_id])
                        for env in vec_envs[other_id].envs:
                            env.opp_policy = avg_policy

            # Plot traj
            for env in vec_envs[agent_id].envs:
                env.ma_env.render_trajectory(os.path.join(save_dir, f"logs_{DEFAULT_ENV}/trajectories_hover/agent{agent_id}_{it:04d}.png"))

    ### SAVE FINAL MODELS ###
    for agent_id in agent_ids:
        models[agent_id].save(os.path.join(save_dir, f"final_agent_{agent_id}_model"))
    print(f"[INFO] Training complete. Models saved in {save_dir}")

    return


    parallel_api_test(env, num_cycles=1_000_000)

def str2bool(val):
    if isinstance(val, bool):
        return val
    if val.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif val.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Single agent reinforcement learning in PyFlyt Gymnasium Environments")
    parser.add_argument('--env',             default=DEFAULT_ENV,             type=str,      help='Single agent gymnasium environment to train (default: hover).')
    parser.add_argument('--retrain',         default=DEFAULT_RETRAIN,         type=str2bool, help='Retrain existing model (default: False).')
    parser.add_argument('--trained_folder',  default=DEFAULT_TRAINED_FOLDER,  type=str,      help='Floder inside output_folder containing model to retrain (default: name)')
    parser.add_argument('--flight_mode',     default=DEFAULT_FLIGHT_MODE,     type=int,      help='Flight mode (0=default).')
    parser.add_argument('--output_folder',   default=DEFAULT_OUTPUT_FOLDER,   type=str,      help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--num_agents',      default=DEFAULT_NUM_AGENTS,      type=int,      help=f'Number of agents in environment (default: {DEFAULT_NUM_AGENTS})')
    parser.add_argument('--total_timesteps', default=DEFAULT_TOTAL_TIMESTEPS, type=int,      help=f'Number of iterations to train agents over (default: {DEFAULT_TOTAL_TIMESTEPS})')
    parser.add_argument('--update_interval', default=DEFAULT_UPDATE_INTERVAL, type=int,      help=f'Intervals for training breaks (default: {DEFAULT_UPDATE_INTERVAL})')
    parser.add_argument('--n_envs',          default=DEFAULT_NUM_ENVS,        type=int,      help=f'Number of environments in vectorized training (default: {DEFAULT_NUM_ENVS})')
    ARGS = parser.parse_args()

    train(**vars(ARGS))
    print("[INFO] Done.")

    

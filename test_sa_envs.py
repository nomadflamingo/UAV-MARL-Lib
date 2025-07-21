import os
import sys
import time
from datetime import datetime
import argparse
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import numpy as np
import torch

# Stable baselines
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy

# Pyflyt 
from PyFlyt.gym_envs.quadx_envs.quadx_hover_env import QuadXHoverEnv
from PyFlyt.gym_envs.quadx_envs.quadx_waypoints_env import QuadXWaypointsEnv

# Global Defaults
ENV_REGISTRY = {
    "hover": QuadXHoverEnv,
    "waypoints": QuadXWaypointsEnv,
}

DEFAULT_ENV = 'hover'
DEFAULT_RETRAIN = False
DEFAULT_FLIGHT_MODE = 0
DEFAULT_OUTPUT_FOLDER = 'results'

# def make_flat_env():
#     env = QuadXWaypointsEnv(render_mode=None, flight_mode=DEFAULT_FLIGHT_MODE)
#     return FlattenObservation(env)

def train(env=DEFAULT_ENV, retrain=DEFAULT_RETRAIN, flight_mode=DEFAULT_FLIGHT_MODE, output_folder=DEFAULT_OUTPUT_FOLDER):
    print(f"\n\n[INFO] Beginning {'re' if retrain else ''}training agents in the \'{env}\' environment.")
    
    # Initiate Environment
    env_class = ENV_REGISTRY[env]
    if env == 'hover':
        policy = 'MlpPolicy'
        target_reward = 500
    elif env == 'waypoints':
        policy = 'MultiInputPolicy'
        target_reward = 380
    else:
        policy = 'MlpPolicy'
        target_reward = 100

    train_env = make_vec_env(
                                env_class,
                                env_kwargs=dict(render_mode=None, flight_mode=flight_mode),
                                n_envs=12, # Increase
                                seed=0
                            )
    eval_env = env_class(render_mode=None, flight_mode=flight_mode)

    print('[INFO] Action Space:', train_env.action_space)
    print('[INFO] Observation Space:', train_env.observation_space)

    if retrain:
        print("[INFO] Loading an model to retrain.")
        filename = './results/save-07.14.2025_22.12.56'
        if not os.path.exists(filename):
            os.makedirs(filename+'/')
        MODEL_PATH = './results/save-07.14.2025_22.12.56/best_model'
        model = SAC.load(MODEL_PATH, env=train_env)

    else:
        print("[INFO] Creating a new model...")
        # Filename
        filename = os.path.join(output_folder, 'save-'+env+'-'+str(flight_mode)+'-'+datetime.now().strftime("%m.%d.%Y_%H.%M"))
        if not os.path.exists(filename):
            os.makedirs(filename+'/')
        
        # Train the model
        model = SAC(policy=policy,
                    env=train_env,
                    tensorboard_log=filename+'/tb/',
                    verbose=1)
    
    # Target cumulative rewards
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=target_reward,
                                                     verbose=1)
    eval_callback = EvalCallback(eval_env,
                                 callback_on_new_best=callback_on_best,
                                 verbose=1,
                                 best_model_save_path=filename+'/',
                                 log_path=filename+'/',
                                 eval_freq=int(5000),
                                 deterministic=True,
                                 render=False)
    model.learn(total_timesteps=int(5e6),
                callback=eval_callback,
                log_interval=500)
    
    #### Save the model ########################################
    model.save(filename+'/final_model.zip')
    print(filename)

    #### Print training progression ############################
    # with np.load(filename+'/evaluations.npz') as data:
    #     for j in range(data['timesteps'].shape[0]):
    #         print(str(data['timesteps'][j])+","+str(data['results'][j][0]))

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
    parser.add_argument('--env',           default=DEFAULT_ENV,           type=str,      help='Single agent gymnasium environment to train (default: hover).')
    parser.add_argument('--retrain',       default=DEFAULT_RETRAIN,       type=str2bool, help='Retrain existing model (default: False).')
    parser.add_argument('--flight_mode',   default=DEFAULT_FLIGHT_MODE,   type=int,      help='Flight mode (0=default).')
    parser.add_argument('--output_folder', default=DEFAULT_OUTPUT_FOLDER, type=str,      help='Folder where to save logs (default: "results")', metavar='')
    ARGS = parser.parse_args()
    train(**vars(ARGS))

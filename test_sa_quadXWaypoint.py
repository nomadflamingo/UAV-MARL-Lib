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
from PyFlyt.gym_envs.quadx_envs.quadx_waypoints_env import QuadXWaypointsEnv

# Global Defaults
DEFAULT_RETRAIN = False
DEFAULT_FLIGHT_MODE = 0
DEFAULT_OUTPUT_FOLDER = 'results'

def make_flat_env():
    env = QuadXWaypointsEnv(render_mode=None, flight_mode=DEFAULT_FLIGHT_MODE)
    return FlattenObservation(env)


def train(retrain=DEFAULT_RETRAIN, flight_mode=DEFAULT_FLIGHT_MODE, output_folder=DEFAULT_OUTPUT_FOLDER):

    # Filename
    filename = os.path.join(output_folder, 'save-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    if not os.path.exists(filename):
        os.makedirs(filename+'/')

    # Initiate Environment

    train_env = make_vec_env(
                                QuadXWaypointsEnv,
                                env_kwargs=dict(render_mode=None, flight_mode=flight_mode),
                                n_envs=1, # Increase
                                seed=0
                            )
    eval_env = QuadXWaypointsEnv(render_mode=None, flight_mode=flight_mode)

    print('[INFO] Action Space:', train_env.action_space)
    print('[INFO] Observation Space:', train_env.observation_space)

    # Train the model
    model = SAC(policy='MultiInputPolicy',
                env=train_env,
                tensorboard_log=filename+'/tb/',
                verbose=1)
    
    # Target cumulative rewards
    target_reward = 400
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=target_reward,
                                                     verbose=1)
    eval_callback = EvalCallback(eval_env,
                                 callback_on_new_best=callback_on_best,
                                 verbose=1,
                                 best_model_save_path=filename+'/',
                                 log_path=filename+'/',
                                 eval_freq=int(1000),
                                 deterministic=True,
                                 render=False)
    model.learn(total_timesteps=int(1e8),
                callback=eval_callback,
                log_interval=100)
    
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
    parser = argparse.ArgumentParser(description="Single agent reinforcement learning for waypoint following")
    parser.add_argument('--retrain',       default=DEFAULT_RETRAIN,       type=str2bool, help='Retrain existing model.')
    parser.add_argument('--flight_mode',   default=DEFAULT_FLIGHT_MODE,   type=int,      help='Flight mode (0=default).')
    parser.add_argument('--output_folder', default=DEFAULT_OUTPUT_FOLDER, type=str,      help='Folder where to save logs (default: "results")', metavar='')
    ARGS = parser.parse_args()
    train(**vars(ARGS))

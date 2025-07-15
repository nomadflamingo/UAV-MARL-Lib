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
from PyFlyt.CL2_envs.quadx_capture_flag import QuadXCaptureFlagEnv
from PyFlyt.CL2_envs.utils.utils import generate_circle_points

# gym_pybullet_drones
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

# Global Defaults
DEFAULT_RETRAIN = False
DEFAULT_FLIGHT_MODE = 0
DEFAULT_OUTPUT_FOLDER = 'results'

def make_flat_env():
    env = QuadXWaypointsEnv(render_mode=None, flight_mode=DEFAULT_FLIGHT_MODE)
    return FlattenObservation(env)


def train(retrain=DEFAULT_RETRAIN, flight_mode=DEFAULT_FLIGHT_MODE, output_folder=DEFAULT_OUTPUT_FOLDER):

    print("\n\n\n[INFO] Starting Demonstration...\n")
    # Filename
    filename = os.path.join(output_folder, 'save-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    if not os.path.exists(filename):
        os.makedirs(filename+'/')

    #### Create the environment ########################################
    env = QuadXCaptureFlagEnv(render_mode="human", flight_mode=7)

    # PYB_CLIENT = env.getPyBulletClient()

    #### Initialze the controller ######################################
    # ctrl = [DSLPIDControl(drone_model=DroneModel("cf2x")) for i in range(env.num_agents)]

    #### Make default waypoints ########################################
    xy_list = generate_circle_points(radius=1.0, n=300)
    start_pos0 = [[float(x), float(y), 0.0, 1.0] for (x, y) in xy_list]
    waypoints = np.array(start_pos0, dtype=np.float32)
    wp_counter = 0
    

    #### Reset Env #####################################################
    obs, _ = env.reset()

    # print(obs)
    # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    # === Main loop ===
    for i in range(15*env.agent_hz):

        obs, rewards, dones, truncs, infos = env.step({
            "uav_0": waypoints[wp_counter%300],
            "uav_1": waypoints[(wp_counter+150)%300],
        })

        # Render frame
        env.render()
        time.sleep(1.0 / env.agent_hz)

        # Exit if either agent is done
        # if any(dones.values()) or any(truncs.values()):
        #     break
        wp_counter += 1

    

def str2bool(val):
    if isinstance(val, bool):
        return val
    if val.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif val.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multi-agent Capture the flag environment")
    parser.add_argument('--retrain',       default=DEFAULT_RETRAIN,       type=str2bool, help='Retrain existing model.')
    parser.add_argument('--flight_mode',   default=DEFAULT_FLIGHT_MODE,   type=int,      help='Flight mode (0=default).')
    parser.add_argument('--output_folder', default=DEFAULT_OUTPUT_FOLDER, type=str,      help='Folder where to save logs (default: "results")', metavar='')
    ARGS = parser.parse_args()
    train(**vars(ARGS))

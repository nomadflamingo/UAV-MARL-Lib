import os
import sys
import time
from datetime import datetime
import argparse
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import numpy as np
import torch
import imageio.v2 as imageio
import pybullet as p

# Stable baselines
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy

# Pyflyt
from PyFlyt.CL2_envs.quadx_capture_flag import QuadXCaptureFlagEnv
from PyFlyt.CL2_envs.utils.utils import generate_circle_points

# gym_pybullet_drones
sys.path.append("/home/nathan/UMASS")
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

# Global Defaults
DEFAULT_FLIGHT_MODE = 7
DEFAULT_OUTPUT_FOLDER = "videos_ctf"

def make_flat_env():
    env = QuadXWaypointsEnv(render_mode=None, flight_mode=DEFAULT_FLIGHT_MODE)
    return FlattenObservation(env)

def make_env(render_mode: str = "human"):
    return QuadXCaptureFlagEnv(render_mode="human", flight_mode=7)

def get_debug_visualizer_frame(env, width=944, height=944): # 944
    img = env.unwrapped.aviary.getCameraImage(
        width=width,
        height=height,
        viewMatrix=env.unwrapped.aviary.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0, 0, 40],
            distance=60.0,
            yaw=45,
            pitch=45,
            roll=0,
            upAxisIndex=2
        ),
        # projectionMatrix=env.unwrapped.aviary.computeProjectionMatrixFOV(
        #     fov=60.0,
        #     aspect=float(width) / height,
        #     nearVal=0.1,
        #     farVal=100.0
        # ),
        # renderer=env.unwrapped.aviary.ER_BULLET_HARDWARE_OPENGL
    )
    rgb_array = np.reshape(img[2], (height, width, 4))[:, :, :3]  # Drop alpha
    return rgb_array

def run(ma_env, output_folder=DEFAULT_OUTPUT_FOLDER, fps=30):

    print("\n\n\n[INFO] Starting Demonstration...\n")
    # Filename
    os.makedirs(output_folder, exist_ok=True)

    #### Create the environment ########################################
    

    #### Initialze the controller ######################################
    # ctrl = [DSLPIDControl(drone_model=DroneModel("cf2x")) for i in range(env.num_agents)]

    #### Make default waypoints ########################################
    num_agents = ma_env.num_possible_agents
    num_waypoints = 300

    xy_list = generate_circle_points(radius=1.0, n=300)
    start_pos0 = [[float(x), float(y), 0.0, 1.0] for (x, y) in xy_list]
    waypoints = np.array(start_pos0, dtype=np.float32)
    wp_counter = 0

    spacing = num_waypoints // num_agents
    

    #### Reset Env #####################################################
    obs, _ = ma_env.reset()

    # print(obs)
    # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    # grab initial camera image
    # frames.append(ma_env.unwrapped.aviary.drones[1].rgbaImg)
    frames = []
    # frames.append(get_debug_visualizer_frame(ma_env))

    # === Main loop ===
    for i in range(6*ma_env.agent_hz):

        actions = {}
        for i, agent in enumerate(ma_env.agents):
            idx = (wp_counter + i * spacing) % num_waypoints
            actions[agent] = waypoints[idx]

        obs, rewards, dones, truncs, infos = ma_env.step(actions)

        # Render frame
        ma_env.render()
        time.sleep(1.0 / ma_env.agent_hz)

        # Get new frame
        frames.append(get_debug_visualizer_frame(ma_env))

        # Exit if either agent is done
        # if any(dones.values()) or any(truncs.values()):
        #     break
        wp_counter += 1

    path = os.path.join(output_folder, f"episode_wide.mp4")
    writer = imageio.get_writer(path, fps=fps, codec='libx264', quality=8)
    for f in frames:
        writer.append_data(f)
    writer.close()

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
    # parser.add_argument('--retrain',       default=DEFAULT_RETRAIN,       type=str2bool, help='Retrain existing model.')
    # parser.add_argument('--flight_mode',   default=DEFAULT_FLIGHT_MODE,   type=int,      help='Flight mode (0=default).')
    # parser.add_argument('--output_folder', default=DEFAULT_OUTPUT_FOLDER, type=str,      help='Folder where to save logs (default: "results")', metavar='')
    # ARGS = parser.parse_args()
    # train(**vars(ARGS))

    env = make_env(render_mode="human")
    run(env)

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import wandb
import time
from datetime import datetime

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.evaluation import evaluate_policy

# Pyflyt
from PyFlyt.gym_envs.quadx_envs.quadx_waypoints_env import QuadXWaypointsEnv

def main():
    print('[INFO] Starting Simulation...')

    # Load Model
    output_folder = 'results'
    filename = 'save-07.09.2025_01.05.40'
    # /home/nathan/Desktop/PyFlyt/results/save-07.09.2025_01.05.40/final_model.zip
    filename = os.path.join(output_folder, filename)
    print(filename)

    if os.path.isfile(filename+'/best_model.zip'):
        path = filename+'/best_model.zip'
    else:
        print("[ERROR]: no model under the specified path", filename)
        exit()

    model = SAC.load(path)

    flight_mode = int(0)

    # Initiate test environment 
    test_env = QuadXWaypointsEnv(render_mode="human", flight_mode=flight_mode)
    test_env_no_gui = QuadXWaypointsEnv(render_mode=None, flight_mode=flight_mode)

    mean_reward, std_reward = evaluate_policy(model,
                                              test_env_no_gui,
                                              n_eval_episodes=10
                                              )
    print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")

    # Simulation
    obs, info = test_env.reset(seed=7)

    while True:
    # for i in range(200):
        action, _states = model.predict(obs,
                                        deterministic=True
                                        )
        obs, reward, terminated, truncated, info = test_env.step(action)

        # print("Obs:", obs, "\tAction:", action, "\tReward:", reward, "\tTerminated:", terminated, "\tTruncated:", truncated)

         # Render frame
        test_env.render()
        time.sleep(1.0 / test_env.agent_hz)

        # Exit if either agent is done
        if terminated or truncated:
            break

    test_env.close()

    # Plot


if __name__ == "__main__":
    main()
    

   
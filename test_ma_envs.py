import os
import wandb
import argparse
import numpy as np
from datetime import datetime
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from pettingzoo.test import parallel_api_test

import gymnasium as gym
from gymnasium import spaces

from PyFlyt.pz_envs import MAFixedwingDogfightEnvV2
from PyFlyt.pz_envs.quadx_envs.ma_combat_env import CombatWaypointPursuitEnv
from PyFlyt.pz_envs.quadx_envs.ma_quadx_hover_env import MAQuadXHoverEnv

from pettingzoo.test import parallel_api_test

import supersuit as ss
from wandb.integration.sb3 import WandbCallback
os.environ["WANDB_MODE"] = "disabled"  

# Global Defaults
ENV_REGISTRY = {
    "dogfight": MAFixedwingDogfightEnvV2,
    "combat": CombatWaypointPursuitEnv,
    "hover": MAQuadXHoverEnv,
}

DEFAULT_ENV = 'combat'
DEFAULT_RETRAIN = False
DEFAULT_FLIGHT_MODE = 0
DEFAULT_OUTPUT_FOLDER = 'results/ma'

class RandomPolicy:
    """A dummy policy that returns random actions from a given action_space."""
    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, obs, deterministic=True):
        # SB3 expects a tuple (action, state)
        return self.action_space.sample(), None
    
class SelfPlayEnv(gym.Env):
    """
    Wraps CombatWaypointPursuitEnv for one agent.
    The training agent takes its action from `train_model.predict()`,
    and the opponent from `opp_policy.predict()`.
    """
    def __init__(self, ma_env, train_agent_id: int, opp_policy):
        super().__init__()
        self.ma_env     = ma_env
        self.train_id   = train_agent_id
        self.opp_id     = 1 - train_agent_id
        self.opp_policy = opp_policy

        # One reset to infer observation shape
        obs_dict, _ = self.ma_env.reset()
        name        = self.ma_env.agents[self.train_id]
        # print(f"OBS_DICT \n", obs_dict)
        sample_obs  = obs_dict[name]
        # print(f"SAMPLE_OBS \n", sample_obs)

        # print(f"SAMPLE_OBS ATT SHAPE \n", sample_obs["attitude"].shape)
        # print(f"SAMPLE_OBS DELT SHAPE \n", sample_obs["target_deltas"].shape)

        # Build SB3-compatible spaces
        if DEFAULT_ENV == 'combat':
            self.observation_space = spaces.Dict({
                "attitude":      spaces.Box(
                                    low=-np.inf,
                                    high=np.inf,
                                    shape=sample_obs["attitude"].shape,
                                    dtype=np.float32),
                "target_deltas": spaces.Box(
                                    low=-np.inf,
                                    high=np.inf,
                                    shape=sample_obs["target_deltas"].shape,
                                    dtype=np.float32),
            })
        else:
            self.observation_space = ma_env.observation_space(name)

        self.action_space = ma_env.action_space(name)

    def reset(self, *, seed=None, options=None):
        if seed is not None or options is not None:
            obs_dict, infos = self.ma_env.reset(seed=seed, options=options)
        else:
            obs_dict, infos = self.ma_env.reset()
        name = self.ma_env.agents[self.train_id]
        return obs_dict[name], infos.get(name, {})

    def step(self, action):
        ag_train = self.ma_env.agents[self.train_id]
        ag_opp   = self.ma_env.agents[self.opp_id]

        # opponent observation
        opp_obs = getattr(self, "_last_obs", {}).get(ag_opp, None)
        if opp_obs is None:
            # fallback to sampling a random action first
            curr_obs, rewards, terms, truncs, infos = self.ma_env.step({
                ag_train: action,
                ag_opp:   self.action_space.sample(),
            })
            opp_obs = curr_obs[ag_opp]
        opp_action, _ = self.opp_policy.predict(opp_obs, deterministic=True)

        # step multi-agent env
        obs_dict, rewards, terms, truncs, infos = self.ma_env.step({
            ag_train: action,
            ag_opp:   opp_action,
        })
        self._last_obs = obs_dict

        

        obs  = obs_dict[ag_train]
        rew  = rewards[ag_train]
        # done = bool(terms[ag_train] or truncs[ag_train])
        terminated = bool(terms[ag_train]   or terms[ag_opp])
        truncated  = bool(truncs[ag_train]  or truncs[ag_opp])
        info = infos.get(ag_train, {})

        if terminated or truncated:
            obs, info = self.reset()
        return obs, rew, terminated, truncated, info

    def render(self, *args, **kwargs):
        return self.ma_env.render(*args, **kwargs)

def make_env(ma_env, train_agent_id: int, seed: int, n_envs: int, flight_mode: int):
    def _init():
        # ma_env = CombatWaypointPursuitEnv(render_mode=None, flight_mode=flight_mode)
        ma_env.reset()
        random_opp = RandomPolicy(ma_env.action_space(ma_env.agents[1 - train_agent_id]))
        env = SelfPlayEnv(ma_env, train_agent_id, random_opp)
        env.reset(seed=seed + train_agent_id)
        return env
    return _init

def train(env=DEFAULT_ENV, 
          retrain=DEFAULT_RETRAIN, 
          flight_mode=DEFAULT_FLIGHT_MODE, 
          output_folder=DEFAULT_OUTPUT_FOLDER, 
          trained_folder='name',
          num_agents=2,
          total_timesteps=int(1e6),
          update_interval=100_000,
          n_envs=8):
    
    print(f"\n\n[INFO] Beginning {'re' if retrain else ''}training agents in the \'{env}\' environment.")
    
    agent_ids = list(range(num_agents))
    agent_names = [f"uav_{i}" for i in range(num_agents)]

    ### INITIATE THE ENVIRONMENTS ###
    env_class = ENV_REGISTRY[env]
    ma_env = env_class(render_mode=None, flight_mode=flight_mode)
    if env == 'hover':
        policy = 'MultiInputPolicy'
        target_reward = 1600
    elif env == 'dogfight':
        policy = 'MultiInputPolicy'
        target_reward = 380
    elif env == 'combat':
        policy = 'MultiInputPolicy'
        target_reward = 380
    else:
        print("[ERROR] This environment is not currently suited to train the environment,", env)
        exit()

    # Create File 
    save_dir = os.path.join(output_folder, 'save-'+env+'-'+str(flight_mode)+'-'+datetime.now().strftime("%m.%d.%Y_%H.%M"))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir+'/')

    # Load or initiate the models
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
        eval_envs[agent_id] = make_env(ma_env, agent_id, seed=1000 + agent_id, n_envs=1, flight_mode=flight_mode)()

        # Train model
        models[agent_id] = SAC(
            policy=policy,
            env=vec_envs[agent_id],
            verbose=1,
            tensorboard_log=os.path.join(save_dir, f"tb_agent_{agent_id}")
        )


    ### CALLBACKS ###
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
            )
        ]

    ### TRAINING LOOP ###
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
            for other_id in agent_ids:
                if other_id != agent_id:
                    for env in vec_envs[other_id].envs:
                        env.opp_policy = models[agent_id]

    ### SAVE FINAL MODELS ###
    for agent_id in agent_ids:
        models[agent_id].save(os.path.join(save_dir, f"final_agent_{agent_id}_model"))
    print(f"[INFO] Training complete. Models saved in {save_dir}")

    return

if __name__ == "__main__":
    env = MAQuadXHoverEnv()
    parallel_api_test(env, num_cycles=1_000_000)

    env = ss.black_death_v3(env)

    env.reset()

    print(f"Starting training on {str(env.metadata['name'])}.")

    env = ss.pettingzoo_env_to_vec_env_v1(env)

    train()
    print("[INFO] Done.")
    
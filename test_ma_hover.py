import os
import argparse
import numpy as np
from datetime import datetime

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from pettingzoo.test import parallel_api_test

from PyFlyt.pz_envs.quadx_envs.ma_combat_env2 import CombatWaypointPursuitEnv2

# Globals
DEFAULT_RETRAIN = False
DEFAULT_FLIGHT_MODE = 0
DEFAULT_OUTPUT_FOLDER = 'results/ma'
DEFAULT_TOTAL_TIMESTEPS = int(5e6)

# ————————————————————————————————————————————————————————————————
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
    def __init__(self, ma_env: CombatWaypointPursuitEnv2, train_agent_id: int, opp_policy):
        super().__init__()
        self.ma_env     = ma_env
        self.train_id   = train_agent_id
        self.opp_id     = 1 - train_agent_id
        self.opp_policy = opp_policy

        # One reset to infer observation shape
        obs_dict, _ = self.ma_env.reset()
        name        = self.ma_env.agents[self.train_id]
        sample_obs  = obs_dict[name]

        # Build SB3-compatible spaces
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

def make_env(train_agent_id: int, seed: int, n_envs: int, flight_mode: int):
    def _init():
        ma_env    = CombatWaypointPursuitEnv2(render_mode=None, flight_mode=flight_mode)
        ma_env.reset()
        random_opp = RandomPolicy(ma_env.action_space(ma_env.agents[1 - train_agent_id]))
        env       = SelfPlayEnv(ma_env, train_agent_id, random_opp)
        env.reset(seed=seed + train_agent_id)
        return env
    return _init

def train(retrain=False, flight_mode=0, output_folder="./results", total_timesteps=int(1e1), update_interval=100_000, n_envs=8):

    # Final save path
    save_dir = os.path.join(output_folder, 'save-selfplay-' + datetime.now().strftime("%m.%d.%Y_%H.%M"))
    os.makedirs(save_dir, exist_ok=True)

    # — Build vectorized envs for ego (agent 0) and adv (agent 1)
    vec_ego = VecMonitor(
        DummyVecEnv([make_env(0, seed=42, n_envs=n_envs, flight_mode=flight_mode) for _ in range(n_envs)])
    )
    vec_adv = VecMonitor(
        DummyVecEnv([make_env(1, seed=4242, n_envs=n_envs, flight_mode=flight_mode) for _ in range(n_envs)])
    )

    if retrain:
        print("[INFO] Loading models to retrain.")
        model_ego = SAC.load(os.path.join(save_dir, "best_ego_model"), env=vec_ego)
        model_adv = SAC.load(os.path.join(save_dir, "best_adv_model"), env=vec_adv)
    else:
        print("[INFO] Training new models.")
        model_ego = SAC("MultiInputPolicy", vec_ego, verbose=1, tensorboard_log=os.path.join(save_dir, "tb_ego"))
        model_adv = SAC("MultiInputPolicy", vec_adv, verbose=1, tensorboard_log=os.path.join(save_dir, "tb_adv"))

    # — Eval environments (used by EvalCallback)
    eval_env_ego = make_env(0, seed=123, n_envs=1, flight_mode=flight_mode)()
    eval_env_adv = make_env(1, seed=321, n_envs=1, flight_mode=flight_mode)()

    # — Callbacks
    eval_cb_ego = EvalCallback(
        eval_env_ego,
        best_model_save_path=os.path.join(save_dir, "ego_eval"),
        log_path=os.path.join(save_dir, "ego_eval"),
        eval_freq=10_000,
        deterministic=True,
        render=False,
    )
    eval_cb_adv = EvalCallback(
        eval_env_adv,
        best_model_save_path=os.path.join(save_dir, "adv_eval"),
        log_path=os.path.join(save_dir, "adv_eval"),
        eval_freq=10_000,
        deterministic=True,
        render=False,
    )
    checkpoint_ego = CheckpointCallback(save_freq=250_000 // n_envs, save_path=os.path.join(save_dir, "checkpoints/ego"), name_prefix="ego")
    checkpoint_adv = CheckpointCallback(save_freq=250_000 // n_envs, save_path=os.path.join(save_dir, "checkpoints/adv"), name_prefix="adv")

    # — Self-play training loop
    n_iters = total_timesteps // update_interval
    for it in range(1, n_iters + 1):
        print(f"[Iter {it}/{n_iters}] ▶ Training Ego")
        model_ego.learn(
            total_timesteps=update_interval,
            reset_num_timesteps=False,
            callback=[eval_cb_ego, checkpoint_ego],
        )
        # broadcast ego → adv
        for env in vec_adv.envs:
            env.opp_policy = model_ego

        print(f"[Iter {it}/{n_iters}] ▶ Training Adversary")
        model_adv.learn(
            total_timesteps=update_interval,
            reset_num_timesteps=False,
            callback=[eval_cb_adv, checkpoint_adv],
        )
        # broadcast adv → ego
        for env in vec_ego.envs:
            env.opp_policy = model_adv

    # Final save
    model_ego.save(os.path.join(save_dir, "final_ego_model"))
    model_adv.save(os.path.join(save_dir, "final_adv_model"))
    print(f"[INFO] Training complete. Models saved in {save_dir}")

def str2bool(val):
    """Converts a string into a boolean.

    Parameters
    ----------
    val : str | bool
        Input value (possibly string) to interpret as boolean.

    Returns
    -------
    bool
        Interpretation of `val` as True or False.

    """
    if isinstance(val, bool):
        return val
    elif val.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif val.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError("[ERROR] in str2bool(), a Boolean value is expected")
    
if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Training script for combat environment')
    parser.add_argument('--retrain',               default=DEFAULT_RETRAIN,               type=str2bool,      help='Loads a previously trained model for more learning (default: False)', metavar='')
    parser.add_argument('--flight_mode',           default=DEFAULT_FLIGHT_MODE,           type=int,           help='Interger defined flight mode for Quadcopter (default: 0 -> vp, vq, vr, T)', metavar='')
    parser.add_argument('--output_folder',         default=DEFAULT_OUTPUT_FOLDER,         type=str,           help='Folder where to save logs (default: "results/ma")')
    parser.add_argument('--total_timesteps',       default=DEFAULT_TOTAL_TIMESTEPS,       type=int,           help='')

    ARGS = parser.parse_args()
    env = CombatWaypointPursuitEnv2()
    parallel_api_test(env, num_cycles=1_000_000)
    try:
        env = CombatWaypointPursuitEnv2()
        parallel_api_test(env, num_cycles=1_000_000)
    except:
        print("[ERROR] There seems to be a problem with the environment.")
        exit()

    train(**vars(ARGS))

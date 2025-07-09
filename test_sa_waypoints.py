import os
import numpy as np
import gymnasium as gym
import wandb
import argparse

from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback
from wandb.integration.sb3 import WandbCallback

from typing import Callable
from gymnasium import Env
from PyFlyt.gym_envs.quadx_envs.quadx_waypoints_env import QuadXWaypointsEnv

DEFAULT_RETRAIN = False
DEFAULT_FLIGHT_MODE = 0

# class RandomPolicy:
#     def __init__(self, action_space):
#         self.action_space = action_space

#     def predict(self, obs, deterministic=True):
#         return self.action_space.sample(), None

# class SelfPlayEnv(gym.Env):
    # def __init__(self, ma_env: CombatWaypointPursuitEnv, train_agent_id: int, opp_policy):
    #     super().__init__()
    #     self.ma_env     = ma_env
    #     self.train_id   = train_agent_id
    #     self.opp_id     = 1 - train_agent_id
    #     self.opp_policy = opp_policy

    #     obs_dict, _ = self.ma_env.reset()
    #     name = self.ma_env.agents[self.train_id]
    #     sample_obs = obs_dict[name]

    #     self.observation_space = spaces.Dict({
    #         "attitude": spaces.Box(low=-np.inf, high=np.inf, shape=sample_obs["attitude"].shape, dtype=np.float32),
    #         "target_deltas": spaces.Box(low=-np.inf, high=np.inf, shape=sample_obs["target_deltas"].shape, dtype=np.float32),
    #     })
    #     self.action_space = ma_env.action_space(name)

    # def reset(self, *, seed=None, options=None):
    #     obs_dict, infos = self.ma_env.reset(seed=seed, options=options)
    #     name = self.ma_env.agents[self.train_id]
    #     return obs_dict[name], infos.get(name, {})

    # def step(self, action):
    #     ag_train = self.ma_env.agents[self.train_id]
    #     ag_opp   = self.ma_env.agents[self.opp_id]

    #     # Get opponent action
    #     opp_obs = getattr(self, "_last_obs", {}).get(ag_opp, None)
    #     if opp_obs is None:
    #         obs_dict, _, _, _, _ = self.ma_env.step({
    #             ag_train: action,
    #             ag_opp: self.action_space.sample(),
    #         })
    #         opp_obs = obs_dict[ag_opp]
    #     opp_action, _ = self.opp_policy.predict(opp_obs, deterministic=True)

    #     obs_dict, rewards, terms, truncs, infos = self.ma_env.step({
    #         ag_train: action,
    #         ag_opp: opp_action,
    #     })
    #     self._last_obs = obs_dict

    #     obs = obs_dict[ag_train]
    #     rew = rewards[ag_train]
    #     terminated = bool(terms[ag_train] or terms[ag_opp])
    #     truncated  = bool(truncs[ag_train] or truncs[ag_opp])
    #     info = infos.get(ag_train, {})

    #     if terminated or truncated:
    #         obs, info = self.reset()
    #     return obs, rew, terminated, truncated, info

def make_env(seed=0, flight_mode=0) -> Callable[[], Env]:
    def _init() -> Env:
        env = QuadXWaypointsEnv(render_mode=None, flight_mode=flight_mode)
        # env.reset(seed=seed)
        return env
    return _init

def train(retrain=DEFAULT_RETRAIN, flight_mode=DEFAULT_FLIGHT_MODE):
    os.environ["WANDB_MODE"] = "disabled"
    wandb.init(
        project="combat_pursuit",
        name="sac_ego_waypoints_only",
        config={
            "total_timesteps": int(1e3),
            "update_interval": 1000,
            "n_envs": 4,
        },
    )

    total_timesteps = wandb.config.total_timesteps
    update_interval = wandb.config.update_interval
    n_envs = wandb.config.n_envs
    n_iters = total_timesteps // update_interval

    # Ego only environment
    env_fns = [make_env(seed=42 + i, flight_mode=flight_mode) for i in range(n_envs)]

    # Debug check
    for i, fn in enumerate(env_fns):
        obs, info = fn().reset()
        print(f"[DEBUG] Env {i} reset -> keys: {list(obs.keys())}, shapes: {[v.shape for v in obs.values()]}")

    vec_ego = DummyVecEnv(env_fns)


    if retrain:
        print("[INFO] Loading pretrained ego model.")
        model_ego = SAC.load('./final_models/ego_sac_waypoints', env=vec_ego)
    else:
        model_ego = SAC(
            policy="MultiInputPolicy",
            env=vec_ego,
            verbose=1,
            tensorboard_log="./tensorboard/ego/",
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=250_000 // n_envs,
        save_path="./checkpoints/ego/",
        name_prefix="ego_sac"
    )
    wandb_cb = WandbCallback(verbose=2, model_save_path=None)

    for it in range(1, n_iters + 1):
        print(f"[Iter {it}/{n_iters}] ▶ Training Ego")
        model_ego.learn(
            total_timesteps=update_interval,
            reset_num_timesteps=False,
            callback=[checkpoint_cb, wandb_cb],
        )

    model_ego.save("./final_models/ego_sac_waypoints")
    wandb.finish()

def str2bool(val):
    if isinstance(val, bool):
        return val
    if val.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif val.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--retrain', default=DEFAULT_RETRAIN, type=str2bool, help='Retrain existing model.')
    parser.add_argument('--flight_mode', default=DEFAULT_FLIGHT_MODE, type=int, help='Flight mode (0=default).')
    ARGS = parser.parse_args()
    train(**vars(ARGS))

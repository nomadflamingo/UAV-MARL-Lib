import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import wandb
import time

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback
from wandb.integration.sb3 import WandbCallback

from PyFlyt.pz_envs import CombatWaypointPursuitEnv

class SelfPlayEnv(gym.Env):
    """
    Wraps CombatWaypointPursuitEnv for one agent.
    The training agent takes its action from `train_model.predict()`,
    and the opponent from `opp_policy.predict()`.
    """
    def __init__(self, ma_env: CombatWaypointPursuitEnv, train_agent_id: int, opp_policy):
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

if __name__ == "__main__":

    EGO_MODEL_PATH = './checkpoints/ego/ego_sac_750000_steps'
    ADV_MODEL_PATH = './checkpoints/adv/adv_sac_750000_steps'

    model_ego = SAC.load(EGO_MODEL_PATH)
    model_adv = SAC.load(ADV_MODEL_PATH)

    # === Initialize environment with rendering enabled ===
    env = CombatWaypointPursuitEnv(render_mode="human")
    obs, _ = env.reset()

    print(obs)
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    # === Main loop ===
    while True:
        ego_obs = obs["uav_0"]
        adv_obs = obs["uav_1"]

        ego_action, _ = model_ego.predict(ego_obs, deterministic=True)
        adv_action, _ = model_adv.predict(adv_obs, deterministic=True)

        obs, rewards, dones, truncs, infos = env.step({
            "uav_0": ego_action,
            "uav_1": adv_action,
        })

        # Render frame
        env.render()
        time.sleep(1.0 / env.agent_hz)

        # Exit if either agent is done
        if any(dones.values()) or any(truncs.values()):
            break

    # Optional: show trajectories
    env.render_trajectory()

   
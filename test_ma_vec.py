import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import wandb
import argparse

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback
from wandb.integration.sb3 import WandbCallback

from PyFlyt.pz_envs import CombatWaypointPursuitEnv

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

def make_env(train_agent_id: int, seed: int, n_envs: int, flight_mode: int):
    def _init():
        ma_env    = CombatWaypointPursuitEnv(render_mode=None, flight_mode=flight_mode)
        ma_env.reset()
        random_opp = RandomPolicy(ma_env.action_space(ma_env.agents[1 - train_agent_id]))
        env       = SelfPlayEnv(ma_env, train_agent_id, random_opp)
        env.reset(seed=seed + train_agent_id)
        return env
    return _init

def train(load_model, flight_mode):
    os.environ["WANDB_MODE"] = "disabled"

    wandb.init(
        project="combat_pursuit",
        name="sac_selfplay_parallel",
        config={
            "total_timesteps": int(1e4),
            "update_interval": 1_000,
            "n_envs": 8,
        },
    )

    total_timesteps  = wandb.config.total_timesteps
    update_interval  = wandb.config.update_interval
    n_envs           = wandb.config.n_envs
    n_iters         = total_timesteps // update_interval

    # — Build vectorized envs for ego (agent 0) and adv (agent 1)
    vec_ego = VecMonitor(
        DummyVecEnv([make_env(0, seed=42, n_envs=n_envs, flight_mode=flight_mode) for _ in range(n_envs)])
    )
    vec_adv = VecMonitor(
        DummyVecEnv([make_env(1, seed=4242, n_envs=n_envs, flight_mode=flight_mode) for _ in range(n_envs)])
    )

    if load_model:
        print("[INFO] Loading an model to retrain.")
        EGO_MODEL_PATH = './checkpoints/ego/ego_sac_750000_steps'
        ADV_MODEL_PATH = './checkpoints/adv/adv_sac_750000_steps'
        model_ego = SAC.load(EGO_MODEL_PATH, env=vec_ego)
        model_adv = SAC.load(ADV_MODEL_PATH, env=vec_adv)
    else:
        print("[INFO] Training a new model.")
        # — Instantiate SAC on each VecEnv
        model_ego = SAC(
            policy="MultiInputPolicy",
            env=vec_ego,
            verbose=1,
            tensorboard_log="./tensorboard/ego/",
        )
        model_adv = SAC(
            policy="MultiInputPolicy",
            env=vec_adv,
            verbose=1,
            tensorboard_log="./tensorboard/adv/",
        )

    # — Callbacks
    checkpoint_ego = CheckpointCallback(
        save_freq=200_000 // n_envs,
        save_path="./checkpoints/ego/",
        name_prefix="ego_sac"
    )
    checkpoint_adv = CheckpointCallback(
        save_freq=200_000 // n_envs,
        save_path="./checkpoints/adv/",
        name_prefix="adv_sac"
    )
    wandb_cb = WandbCallback(verbose=2, model_save_path=None)

    # — Self-play training loop
    for it in range(1, n_iters + 1):
        print(f"[Iter {it}/{n_iters}] ▶ Training Ego")
        model_ego.learn(
            total_timesteps=update_interval,
            reset_num_timesteps=False,
            callback=[checkpoint_ego, wandb_cb],
        )
        # broadcast latest ego to adv envs
        for env in vec_adv.envs:
            env.opp_policy = model_ego

        print(f"[Iter {it}/{n_iters}] ▶ Training Adversary")
        model_adv.learn(
            total_timesteps=update_interval,
            reset_num_timesteps=False,
            callback=[checkpoint_adv, wandb_cb],
        )
        # broadcast latest adv to ego envs
        for env in vec_ego.envs:
            env.opp_policy = model_adv

    # — Final save
    model_ego.save("./final_models/ego_sac_parallel")
    model_adv.save("./final_models/adv_sac_parallel")
    wandb.finish()


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
    parser = argparse.ArgumentParser(description='Flight script using CtrlAviary and Model Predictive Control')
    parser.add_argument('--retrain',               default=False,               type=str2bool,      help='Loads a previously trained model for more learning (default: False)', metavar='')
    parser.add_argument('--flight_mode',           default=0,                   type=int,           help='Interger defined flight mode for Quadcopter (default: 0 -> vp, vq, vr, T)', metavar='')

    ARGS = parser.parse_args()

    train(**vars(ARGS))

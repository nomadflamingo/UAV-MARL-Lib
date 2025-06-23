import gymnasium as gym
import numpy as np
from gymnasium import spaces

class SelfPlayEnv(gym.Env):
    """
    Wraps CombatWaypointPursuitEnv for one agent. 
    The training agent takes its action from `train_model.predict()`,
    and the opponent from `opp_policy.predict()`.
    """
    def __init__(self, ma_env, train_agent_id, opp_policy):
        super().__init__()
        self.ma_env      = ma_env
        self.train_id    = train_agent_id
        self.opp_id      = 1 - train_agent_id
        self.opp_policy  = opp_policy

        # One reset to infer observation shape
        obs_dict, _ = self.ma_env.reset()
        name = self.ma_env.agents[self.train_id]
        sample_obs = obs_dict[name]

        # Build SB3-compatible spaces
        self.observation_space = spaces.Dict({
            "attitude":      spaces.Box(
                                  low=-np.inf, 
                                  high=np.inf,
                                  shape=sample_obs["attitude"].shape,
                                  dtype=np.float64),
            "target_deltas": spaces.Box(
                                  low=-np.inf,
                                  high=np.inf,
                                  shape=sample_obs["target_deltas"].shape,
                                  dtype=np.float64),
        })
        self.action_space = ma_env.action_space(name)

    def reset(self, *, seed=None, options=None):
        # Forward seed/options so SB3 can drive resets
        if seed is not None or options is not None:
            obs_dict, infos = self.ma_env.reset(seed=seed, options=options)
        else:
            obs_dict, infos = self.ma_env.reset()
        name = self.ma_env.agents[self.train_id]
        return obs_dict[name], infos.get(name, {})

    def step(self, action):
        # Build the multi-agent action dict
        ag_train = self.ma_env.agents[self.train_id]
        ag_opp   = self.ma_env.agents[self.opp_id]

        # Query opponent policy (fixed during this .learn() call)
        opp_obs = self._last_obs[ag_opp] if hasattr(self, "_last_obs") else None
        # In practice SB3 ensures _last_obs is set; if not, fallback to current step:
        if opp_obs is None:
            print("Warning: _last_obs not set, using current step observation.")
            curr_obs, rewards, terms, truncs, infos = self.ma_env.step({ag_train: action, ag_opp: self.action_space.sample()})
            opp_obs = curr_obs[ag_opp]
        opp_action, _ = self.opp_policy.predict(opp_obs, deterministic=True)

        # Step the MA env
        obs_dict, rewards, terms, truncs, infos = self.ma_env.step({
            ag_train: action,
            ag_opp:   opp_action
        })

        # Save for next opponent call
        self._last_obs = obs_dict

        # Extract this agent’s data
        obs  = obs_dict[ag_train]
        rew  = rewards[ag_train]
        done = bool(terms[ag_train] or truncs[ag_train])
        info = infos.get(ag_train, {})

        if done:
            # immediately reset so SB3’s DummyVecEnv can continue
            obs, info = self.reset()

        return obs, rew, truncs[ag_train], truncs[ag_train], info

    def render(self, *args, **kwargs):
        return self.ma_env.render(*args, **kwargs)



class SelfPlayEnvWings(gym.Env):
    """
    Wraps CombatWaypointPursuitEnv for one agent. 
    The training agent takes its action from `train_model.predict()`,
    and the opponent from `opp_policy.predict()`.
    """
    def __init__(self, ma_env, train_agent_id, opp_policy):
        super().__init__()
        self.ma_env      = ma_env
        self.train_id    = train_agent_id
        self.opp_id      = 1 - train_agent_id
        self.opp_policy  = opp_policy

        # One reset to infer observation shape
        obs_dict, _ = self.ma_env.reset()
        name = self.ma_env.agents[self.train_id]
        sample_obs = obs_dict[name]

        # Build SB3-compatible spaces
        self.observation_space = ma_env.observation_space()
        self.action_space = ma_env.action_space()

    def reset(self, *, seed=None, options=None):
        # Forward seed/options so SB3 can drive resets
        if seed is not None or options is not None:
            obs_dict, infos = self.ma_env.reset(seed=seed, options=options)
        else:
            obs_dict, infos = self.ma_env.reset()
        name = self.ma_env.agents[self.train_id]
        return obs_dict[name], infos.get(name, {})

    def step(self, action):
        # Build the multi-agent action dict
        ag_train = self.ma_env.agents[self.train_id]
        ag_opp   = self.ma_env.agents[self.opp_id]

        # Query opponent policy (fixed during this .learn() call)
        opp_obs = self._last_obs[ag_opp] if hasattr(self, "_last_obs") else None
        # In practice SB3 ensures _last_obs is set; if not, fallback to current step:
        if opp_obs is None:
            print("Warning: _last_obs not set, using current step observation.")
            curr_obs, rewards, terms, truncs, infos = self.ma_env.step({ag_train: action, ag_opp: self.action_space.sample()})
            opp_obs = curr_obs[ag_opp]
        opp_action, _ = self.opp_policy.predict(opp_obs, deterministic=True)

        # Step the MA env
        obs_dict, rewards, terms, truncs, infos = self.ma_env.step({
            ag_train: action,
            ag_opp:   opp_action
        })

        # Save for next opponent call
        self._last_obs = obs_dict

        # Extract this agent’s data
        obs  = obs_dict[ag_train]
        rew  = rewards[ag_train]
        done = bool(terms[ag_train] or truncs[ag_train])
        info = infos.get(ag_train, {})

        if done:
            # immediately reset so SB3’s DummyVecEnv can continue
            obs, info = self.reset()

        return obs, rew, terms[ag_train], truncs[ag_train], info

    def render(self, *args, **kwargs):
        return self.ma_env.render(*args, **kwargs)
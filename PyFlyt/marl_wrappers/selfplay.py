import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Any, Literal

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
        sample_obs  = obs_dict[name]

        # Build SB3-compatible spaces
        if ma_env.metadata["name"] == 'combat_pursuit':
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

class MASelfPlayEnv(gym.Env):
    """
    Wraps a multi-agent PettingZoo env to train one agent against multiple opponents.
    `train_agent_id` is the index of the agent being trained.
    `opp_policies` is a dict mapping agent index to a fixed opponent policy.
    """
    def __init__(self, ma_env, train_agent_id: int, opp_policies: dict[int, Any]):
        super().__init__()
        self.ma_env = ma_env
        self.train_id = train_agent_id
        self.opp_ids = [i for i in range(len(ma_env.agents)) if i != train_agent_id]
        self.opp_policies = opp_policies  # dict of {agent_id: policy}

        # One reset to get observation shape
        obs_dict, _ = self.ma_env.reset()
        train_name = self.ma_env.agents[self.train_id]
        sample_obs = obs_dict[train_name]

        # SB3 observation space
        if ma_env.metadata["name"] == 'combat_pursuit':
            self.observation_space = spaces.Dict({
                "attitude": spaces.Box(low=-np.inf, high=np.inf, shape=sample_obs["attitude"].shape, dtype=np.float32),
                "target_deltas": spaces.Box(low=-np.inf, high=np.inf, shape=sample_obs["target_deltas"].shape, dtype=np.float32),
            })
        else:
            self.observation_space = ma_env.observation_space(train_name)

        self.action_space = ma_env.action_space(train_name)

    def reset(self, *, seed=None, options=None):
        obs_dict, infos = self.ma_env.reset(seed=seed, options=options)
        self._last_obs = obs_dict
        train_name = self.ma_env.agents[self.train_id]
        return obs_dict[train_name], infos.get(train_name, {})

    def step(self, action):
        actions = {}

        # Add training agent's action
        train_name = self.ma_env.agents[self.train_id]
        actions[train_name] = action

        # Add each opponent's action
        for opp_id in self.opp_ids:
            opp_name = self.ma_env.agents[opp_id]
            last_obs = self._last_obs.get(opp_name, None)
            if last_obs is None:
                last_obs = self.ma_env.observation_space(opp_name).sample()
            opp_policy = self.opp_policies.get(opp_id)
            if opp_policy is None:
                opp_action = self.ma_env.action_space(opp_name).sample()
            else:
                opp_action, _ = opp_policy.predict(last_obs, deterministic=True)
            actions[opp_name] = opp_action

        # Step environment
        obs_dict, rewards, terms, truncs, infos = self.ma_env.step(actions)
        self._last_obs = obs_dict

        obs = obs_dict[train_name]
        rew = rewards[train_name]
        terminated = any(terms.values())
        truncated = any(truncs.values())
        info = infos.get(train_name, {})

        if terminated or truncated:
            obs, info = self.reset()
        return obs, rew, terminated, truncated, info

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
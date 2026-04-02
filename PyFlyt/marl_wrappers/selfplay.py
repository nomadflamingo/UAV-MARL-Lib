import gymnasium as gym
import numpy as np
import wandb
import copy
import os
from gymnasium import spaces
from typing import Any, Literal

# SB3
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from wandb.integration.sb3 import WandbCallback

# Envs
from PyFlyt.pz_envs import MAFixedwingDogfightEnvV2
from PyFlyt.pz_envs.quadx_envs.ma_combat_env import CombatWaypointPursuitEnv
from PyFlyt.pz_envs.quadx_envs.ma_quadx_hover_env import MAQuadXHoverEnv
from PyFlyt.pz_envs.quadx_envs.ma_quadx_dogfight_env import MAQuadXDogfightEnv

from stable_baselines3.common.callbacks import BaseCallback
# from sb3_contrib.common.wandb_callback import WandbCallback  # if using SB3 contrib

class SelfPlayEnv(gym.Env):
    """
    Wraps ENV for one agent.
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
        elif ma_env.metadata["name"] == 'ma_fixedwing_team_dogfight':
            self.observation_space = ma_env.observation_space()
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
        self.opp_ids = [i for i in range(ma_env.num_possible_agents) if i != train_agent_id]
        self.opp_policies = opp_policies  # dict of {agent_id: policy}

        # One reset to get observation shape
        obs_dict, _ = self.ma_env.reset()
        # print(f"#########\n {ma_env.observation_space} \n #############")
        # exit()
        train_name = self.ma_env.agents[self.train_id]
        sample_obs = obs_dict[train_name]

        print(f'[INFO] Using environment, {ma_env.metadata["name"]}')
        # SB3 observation space
        if ma_env.metadata["name"] == 'combat_pursuit':
            self.observation_space = spaces.Dict({
                "attitude": spaces.Box(low=-np.inf, high=np.inf, shape=sample_obs["attitude"].shape, dtype=np.float32),
                "target_deltas": spaces.Box(low=-np.inf, high=np.inf, shape=sample_obs["target_deltas"].shape, dtype=np.float32),
            })
        elif ma_env.metadata["name"] == 'ma_fixedwing_team_dogfight':
            print("[INFO] Setting 'ma_fixedwing_team_dogfight' shape.")
            self.observation_space = ma_env.observation_space()
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

class FictitiousPlayEnv:
    """
    Wraps a multi-agent PettingZoo env to train one agent against multiple opponents.
    `train_agent_id` is the index of the agent being trained.
    `opp_policies` is a dict mapping agent index to a fixed opponent policy.
    """
    def __init__(self, ma_env, agent_ids, strategy, save_dir, sac_kwargs):
        # super().__init__()
        self.ma_env = ma_env(render_mode=None)
        self.agent_ids = agent_ids
        self.save_dir = save_dir
        self.sac_kwargs = sac_kwargs
        self.models = {i: [] for i in agent_ids}        # List of past BRs
        self.current_br = {}                            # Best response agent (current training model)
        self.policy_dist = {i: [1.0] for i in agent_ids}
        self.strategies = {i: strategy for i in agent_ids} # Distribuition Update Strategy

    def make_env(self, ma_env, train_agent_id: int, seed, n_envs: int):
        def _init():
            print(f"\n[INFO] Evaluation Env for agent_{train_agent_id}: {ma_env}")

            ma_env.reset()

            opp_policies = {
                i: RandomPolicy(ma_env.action_space(i))
                for i in range(ma_env.num_possible_agents) if i != train_agent_id
            }

            env = MASelfPlayEnv(ma_env, train_agent_id, opp_policies)
            env.reset(seed=seed)
            return env
        return _init

    def reset(self, *, seed=None, options=None):
        obs_dict, infos = self.ma_env.reset(seed=seed, options=options)
        self._last_obs = obs_dict
        train_name = self.ma_env.agents[self.train_id]
        return obs_dict[train_name], infos.get(train_name, {})

    def update_policy_distribution(self, agent_id):
        strategy = self.strategies[agent_id]
        policy_avg = self.models[agent_id]
        k = len(policy_avg) # current timestep

        # print(f"[INFO] agent{agent_id} using {strategy} policy update.")
        if strategy.endswith('vp'):
            self.policy_dist[agent_id] = [0.0] * (k-1) + [1.0]
        elif strategy.endswith('fp'):
            print("here")
            if k <= 1: 
                normalized_policy_dist = [1.0]
            else: 
                avg_policy_weight = (k-1) / (k+1)
                new_policy_weight = 2 / (k+1)

                scaled_latest_prob = (1 / avg_policy_weight) * new_policy_weight
                new_policy_dist = self.policy_dist[agent_id] + [scaled_latest_prob]

                total_sum = sum(new_policy_dist)
                normalized_policy_dist = [p / total_sum for p in new_policy_dist]

            self.policy_dist[agent_id] = normalized_policy_dist
        elif strategy.endswith('dp'):
            n = min(k, 10)
            self.policy_dist[agent_id] = [0.0] * (k - n) + [1.0 / n] * n

        if k != len(self.policy_dist[agent_id]):
            print(f"[WARNING] Descrepancy between number of models ({k}) and probability distribution ({len(self.policy_dist[agent_id])}).")
    
    def sample_avg_policy(self, agent_id):
        class AvgPolicy:
            def __init__(self, models, probs, env):
                self.models = models
                self.probs = probs
                self.env = env
                # print(f"[INFO] Probs {self.probs}")
            def predict(self, obs, deterministic=True):
                if len(self.models) == 0:
                    # print(f"[INFO] Agent {agent_id} does not have any models yet, defaulting to RandomPolicy")
                    model = RandomPolicy(self.env.action_space(agent_id))
                else:
                    model = np.random.choice(self.models, p=self.probs)
                return model.predict(obs, deterministic)
        return AvgPolicy(self.models[agent_id], self.policy_dist[agent_id], self.ma_env)
    
    def train_agent(self, agent_id, wb, total_timesteps, callbacks):
        # Create training env using current avg opponent
        opp_ids = [i for i in range(self.ma_env.num_possible_agents) if i != agent_id]
        # Use self-play (train against an agent's own policies) or general play (train against opponents' policies)
        if self.strategies[agent_id][0] == 's': 
            # Self-Play
            opp_policies = {
                    opp_id: self.sample_avg_policy(agent_id)
                    for opp_id in opp_ids
                }
        else:
            opp_policies = {
                    opp_id: self.sample_avg_policy(opp_id)
                    for opp_id in opp_ids
                }

        # train_env = MASelfPlayEnv(self.ma_env, agent_id, opp_policies)
        train_env = VecMonitor(
            DummyVecEnv([
                self.make_env(self.ma_env, agent_id, seed=None, n_envs=1)
            ])
        )


        
        if self.models[agent_id]:
            # print(f"[INFO] Continuing training for agent {agent_id}")
            # Restore previous SAC model (we assume self.current_br stores it)
            model = self.current_br[agent_id]
            model.set_env(train_env)  # Update environment (if it changed)
        else:
            print(f"[INFO] Initializing new model for agent {agent_id}")
            log_dir = os.path.join(self.save_dir, f"tb_logs/agent_{agent_id}")

            sac_kwargs = {**self.sac_kwargs, "tensorboard_log": log_dir}
            model = SAC(env=train_env, **sac_kwargs)

        # Train
        print("[INFO] Training model...")
        model.learn(total_timesteps=total_timesteps, 
                    callback=callbacks,
                    tb_log_name=f"fsp_agent{agent_id}" )
        # model.learn(
        #     total_timesteps=total_timesteps,
        #     callback=WandbCallback(
        #         gradient_save_freq=100,
        #         model_save_path=f"models/{wandb.run.id}",
        #         log="all",          # log gradients, parameters, rewards
        #     )
        # )
        # model.learn(total_timesteps=100000, callback=MyWandbCallback())
        print("[INFO] Training Complete.")
        wb.log({
            "agent": agent_id,
            "fsp_iteration": len(self.models[agent_id]) + 1,
            "timesteps_trained": total_timesteps
        })
        self.current_br[agent_id] = model
        policy_copy = copy.deepcopy(model.policy)
        self.models[agent_id].append(policy_copy)

        self.update_policy_distribution(agent_id)

        # Save
        model.save(os.path.join(self.save_dir, f"agent_{agent_id}_br_{len(self.models[agent_id]):04d}"))


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
    
class RandomPolicy:
    """A dummy policy that returns random actions from a given action_space."""
    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, obs, deterministic=True):
        # SB3 expects a tuple (action, state)
        return self.action_space.sample(), None
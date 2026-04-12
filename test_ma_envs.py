import os
import yaml
import wandb
import argparse
import numpy as np
from datetime import datetime
# SB3
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import CallbackList
os.environ["WANDB_MODE"] = "online"
# Gym
import gymnasium as gym
from gymnasium import spaces
# Envs
from PyFlyt.pz_envs import MAFixedwingDogfightEnvV2
from PyFlyt.pz_envs.quadx_envs.ma_combat_env import CombatWaypointPursuitEnv
from PyFlyt.pz_envs.quadx_envs.ma_quadx_hover_env import MAQuadXHoverEnv
from PyFlyt.pz_envs.quadx_envs.ma_quadx_dogfight_env import MAQuadXDogfightEnv
from PyFlyt.pz_envs.quadx_envs.ma_quadx_pursuit_evasion_env import MAQuadXPursuitEvasionEnv
# pz
from pettingzoo.test import parallel_api_test
# SP
from PyFlyt.marl_wrappers.selfplay import MASelfPlayEnv, FictitiousPlayEnv #, SelfPlayEnv


# Environment registry — maps config "env" keys to PettingZoo env classes
ENV_REGISTRY = {
    "dogfight_FW": MAFixedwingDogfightEnvV2,
    "dogfight_QX": MAQuadXDogfightEnv,       # Implementation not finished
    "combat": CombatWaypointPursuitEnv,      # Results Questionable
    "hover": MAQuadXHoverEnv,
    "pursuit_evasion": MAQuadXPursuitEvasionEnv,
}

# Fallback defaults (used when a config key is absent)
DEFAULT_OUTPUT_FOLDER = 'junk'
DEFAULT_TOTAL_TIMESTEPS = int(5e3)
DEFAULT_UPDATE_INTERVAL = int(1e3)
DEFAULT_NUM_ENVS = 8
DEFAULT_STRAT = 'sdp'

class MyWandbCallback(WandbCallback):
    def _on_step(self) -> bool:
        # Print some info every 1000 steps
        if self.num_timesteps % 1000 == 0:
            # self.locals is a dict with local variables from training
            rewards = self.locals.get('rewards', None)
            print(f"[WandbCallback] Step {self.num_timesteps}, Last reward: {rewards}")

        # Make sure we still call the parent method to log to wandb
        return super()._on_step()

class RewardLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        
        # Grab keys from the first env (assuming all envs have the same keys)
        if "reward_components" in infos[0]:
            for key in infos[0]["reward_components"].keys():
                # Average the component across all parallel envs
                avg_val = np.mean([
                    info["reward_components"][key] 
                    for info in infos if "reward_components" in info
                ])
                self.logger.record(f"reward_components/{key}", avg_val)
                
        return True

class RandomPolicy:
    """A dummy policy that returns random actions from a given action_space."""
    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, obs, deterministic=True):
        # SB3 expects a tuple (action, state)
        return self.action_space.sample(), None
    
##### TRAINING FUNCTION #################################################################################
def train(config: dict):
    """
    Modular training routine driven by a YAML config dict.

    Required top-level keys: env, strategy, total_timesteps, update_interval.
    Optional: policy, n_envs, output_folder, sac_hyperparams, env_params.
    env_params are forwarded as **kwargs to the environment constructor.
    """
    env_name        = config["env"]
    strategy        = config.get("strategy", DEFAULT_STRAT)
    total_timesteps = config.get("total_timesteps", DEFAULT_TOTAL_TIMESTEPS)
    update_interval = config.get("update_interval", DEFAULT_UPDATE_INTERVAL)
    n_envs          = config.get("n_envs", DEFAULT_NUM_ENVS)
    output_folder   = config.get("output_folder", DEFAULT_OUTPUT_FOLDER)
    policy          = config.get("policy", "MlpPolicy")
    sac_hyperparams = config.get("sac_hyperparams") or {}
    env_params      = config.get("env_params") or {}

    print(f"\n\n[INFO] Beginning training agents in the '{env_name}' environment using a '{strategy}' method.")

    #################################
    ### INITIATE THE ENVIRONMENTS ###
    #################################
    if env_name not in ENV_REGISTRY:
        print(f"[ERROR] Unknown environment '{env_name}'. Available: {list(ENV_REGISTRY.keys())}")
        exit(1)

    env_class = ENV_REGISTRY[env_name]
    ma_env = env_class(render_mode=None, **env_params)

    # Use the env's actual agent count — overrides --num_agents
    num_agents = ma_env.num_possible_agents
    agent_ids = list(range(num_agents))
    agent_names = [f"uav_{i}" for i in range(num_agents)]

    # Create File
    save_dir = os.path.join(output_folder, 'save-'+env_name+'-'+datetime.now().strftime("%m.%d.%Y_%H.%M"))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir+'/')

    wb = wandb.init(
        project=f"overnight-{env_name}-project",
        name=f"sac-{env_name}-{strategy}-{datetime.now().strftime('%m.%d.%Y_%H.%M')}",
        sync_tensorboard=True,
        config={
            "algo": "SAC",
            "env": env_name,
            "strategy": strategy,
            "num_agents": num_agents,
            "total_timesteps": total_timesteps,
            "update_interval": update_interval,
            **sac_hyperparams,
        },
    )


    #################################
    ###   BREAK FOR FSP TESTING   ###
    #################################
    sac_kwargs = dict(
        policy=policy,
        verbose=1,
        tensorboard_log=os.path.join(save_dir, "tb_logs"),
        **sac_hyperparams
    )

    Trainer = FictitiousPlayEnv(env_class, agent_ids, strategy, save_dir, sac_kwargs)

    eval_envs = {}
    for agent_id in agent_ids:
        eval_envs[agent_id] = VecMonitor(
            DummyVecEnv([
                Trainer.make_env(ma_env, agent_id, seed=1000+agent_id, n_envs=1)
            ])
        )

    # #################################
    # ###         CALLBACKS         ###
    # #################################
    # wandb_callback = MyWandbCallback(
    #     verbose=2,
    #     model_save_path=None,
    #     log="all",
    #     gradient_save_freq=100,
    #     # sync_tensorboard=True,  # <- important if you log via self.logger.record
    # )

    callbacks = {}
    for agent_id in agent_ids:
        callbacks[agent_id] = CallbackList([
            EvalCallback(
                eval_envs[agent_id],
                best_model_save_path=os.path.join(save_dir, f"eval_agent_{agent_id}"),
                log_path=os.path.join(save_dir, f"eval_agent_{agent_id}"),
                eval_freq=1_000,
                deterministic=True,
                render=False,
            ),
            CheckpointCallback(
                save_freq=1_000,
                save_path=os.path.join(save_dir, f"checkpoints/agent_{agent_id}"),
                name_prefix=f"agent_{agent_id}"
            ),
            RewardLoggingCallback(),
            MyWandbCallback(verbose=0),
        ])

    #################################
    ###       TRAINING LOOP       ###
    #################################
    n_iters = total_timesteps // update_interval
    for it in range(1, n_iters + 1):
        for agent_id in agent_ids:
            print(f"\n[{strategy.upper()} Iter {it}/{n_iters}] ▶ Training Agent {agent_id}")

            if strategy == "double_oracle":
                models[agent_id].learn(
                    total_timesteps=update_interval,
                    reset_num_timesteps=False,
                    callback=callbacks[agent_id]
                )
                for other_id in agent_ids:
                    if other_id != agent_id:
                        for env in vec_envs[other_id].envs:
                            env.opp_policy = models[agent_id]
            else:
                # Training
                Trainer.train_agent(agent_id, wb, total_timesteps=update_interval, callbacks=callbacks[agent_id])
                # wandb.finish()
                # exit()

                # # Exploitability
                # exp_opp, exp_ego, exploitability, sum_exploitability = play_env.compute_exploitability()
                # Trainer.logger.log({
                #     "fsp_iteration": n_iters + 1,
                #     "exploitability/total": exploitability,
                #     "exploitability/mean": sum_exploitability,
                #     "exploitability/opp": exp_opp,
                #     "exploitability/ego": exp_ego,
                # })

                # print(f"Exploitability after FSP Iteration {n_iters + 1}: {exploitability:.4f}")
                # print(f"Mean Exploitability: {sum_exploitability:.4f}")

            # # Plot traj
            # for env in vec_envs[agent_id].envs:
            #     env.ma_env.render_trajectory(os.path.join(save_dir, f"logs_{DEFAULT_ENV}/trajectories_hover/agent{agent_id}_{it:04d}.png"))

    ### SAVE FINAL MODELS ###
    for agent_id in agent_ids:
        if agent_id not in Trainer.current_br:
            continue
        file = os.path.join(save_dir, f"final_agent_{agent_id}_model")
        Trainer.current_br[agent_id].save(file)
        wb.save(file)
    
    wandb.finish()
    print(f"[INFO] Training complete. Models saved in {save_dir}")

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multi-agent UAV training in PyFlyt environments")
    parser.add_argument('--config', required=True, type=str,
                        help='Path to YAML config file (e.g. configs/pursuit_evasion.yaml)')
    # Optional CLI overrides — these take precedence over config file values when provided
    parser.add_argument('--total_timesteps', default=None, type=int,
                        help='Override total_timesteps from config')
    parser.add_argument('--update_interval', default=None, type=int,
                        help='Override update_interval from config')
    parser.add_argument('--output_folder',   default=None, type=str,
                        help='Override output_folder from config')
    parser.add_argument('--strategy',        default=None, type=str,
                        help='Override strategy from config')
    ARGS = parser.parse_args()

    with open(ARGS.config, 'r') as f:
        config = yaml.safe_load(f)

    # Apply CLI overrides
    for key in ('total_timesteps', 'update_interval', 'output_folder', 'strategy'):
        val = getattr(ARGS, key)
        if val is not None:
            config[key] = val

    train(config)
    print("[INFO] Done.")

    

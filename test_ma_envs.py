import os
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


# Global Defaults
ENV_REGISTRY = {
    "dogfight_FW": MAFixedwingDogfightEnvV2,
    "dogfight_QX": MAQuadXDogfightEnv,       # Implementation not finished
    "combat": CombatWaypointPursuitEnv,      # Results Questionable
    "hover": MAQuadXHoverEnv,
    "pursuit_evasion": MAQuadXPursuitEvasionEnv,
}
STRAT_REGISTRY = ['vp',     # Vanilla Play
                  'fp',     # Fictitious Play
                  'dp',     # Delta-Uniform Play
                  'svp',    # Vanilla Self-Play
                  'sfp',    # Fictitious Self-Play
                  'sdp']    # Delta-Uniform Self-Play

DEFAULT_ENV = 'dogfight_FW'
DEFAULT_RETRAIN = False
DEFAULT_TRAINED_FOLDER = 'name'
DEFAULT_FLIGHT_MODE = 0
# DEFAULT_OUTPUT_FOLDER = 'results/ma'
DEFAULT_OUTPUT_FOLDER = 'junk'
DEFAULT_NUM_AGENTS = 2
DEFAULT_TOTAL_TIMESTEPS = int(5e3)
DEFAULT_UPDATE_INTERVAL = int(1e3)
DEFAULT_NUM_ENVS = 8
DEFAULT_STRAT = STRAT_REGISTRY[5]

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
        for info in infos:
            if "reward_components" in info:
                for key, value in info["reward_components"].items():
                    self.logger.record(f"reward_components/{key}", value)
        return True

class RandomPolicy:
    """A dummy policy that returns random actions from a given action_space."""
    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, obs, deterministic=True):
        # SB3 expects a tuple (action, state)
        return self.action_space.sample(), None
    
##### TRAINING FUNCTION #################################################################################
def train(env=DEFAULT_ENV,
          retrain=DEFAULT_RETRAIN,
          trained_folder=DEFAULT_TRAINED_FOLDER,
          flight_mode=DEFAULT_FLIGHT_MODE,
          output_folder=DEFAULT_OUTPUT_FOLDER,
          num_agents=DEFAULT_NUM_AGENTS,
          total_timesteps=DEFAULT_TOTAL_TIMESTEPS,
          update_interval=DEFAULT_UPDATE_INTERVAL,
          n_envs=DEFAULT_NUM_ENVS,
          strategy=DEFAULT_STRAT,
          policy_type="SAC",
          sac_hyperparams: dict | None = None):
    """
    Modular training routine for a choice of "ENV_REGISTRY" and "STRAT_REGISTRY".
    """
    print(f"\n\n[INFO] Beginning {'re' if retrain else ''}training agents in the \'{env}\' environment using a \'{strategy}\' method.")

    #################################
    ### INITIATE THE ENVIRONMENTS ###
    #################################
    env_class = ENV_REGISTRY[env]
    if env == 'hover':
        policy = 'MlpPolicy'
        target_reward = 1600
        ma_env = env_class(render_mode=None, flight_mode=flight_mode)
    elif env == 'dogfight_FW':
        policy = 'MlpPolicy'
        ma_env = env_class(render_mode=None)
    elif env == 'combat':
        policy = 'MultiInputPolicy'
        ma_env = env_class(render_mode=None, flight_mode=flight_mode)
    elif env == 'pursuit_evasion':
        policy = 'MlpPolicy'
        ma_env = env_class(render_mode=None)
    else:
        print("[ERROR] This environment is not currently suited to train the environment,", env)
        exit()

    # Use the env's actual agent count — overrides --num_agents
    num_agents = ma_env.num_possible_agents
    agent_ids = list(range(num_agents))
    agent_names = [f"uav_{i}" for i in range(num_agents)]

    # Create File
    save_dir = os.path.join(output_folder, env)
    save_dir = os.path.join(output_folder, 'save-'+env+'-'+str(flight_mode)+'-'+datetime.now().strftime("%m.%d.%Y_%H.%M"))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir+'/')

    sac_hparams = sac_hyperparams or {}

    wb = wandb.init(
        project=f"overnight-{env}-project",
        name=f"sac-{env}-{strategy}-{datetime.now().strftime('%m.%d.%Y_%H.%M')}",
        config={
            "algo": "SAC",
            "env": env,
            "strategy": strategy,
            "num_agents": num_agents,
            "total_timesteps": total_timesteps,
            "update_interval": update_interval,
            **sac_hparams,
        },
    )


    #################################
    ###   BREAK FOR FSP TESTING   ###
    #################################
    sac_kwargs = dict(policy=policy, verbose=1, **sac_hparams)

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


def str2bool(val):
    if isinstance(val, bool):
        return val
    if val.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif val.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Single agent reinforcement learning in PyFlyt Gymnasium Environments")
    parser.add_argument('--env',             default=DEFAULT_ENV,             type=str,      help='Single agent gymnasium environment to train (default: hover).')
    parser.add_argument('--retrain',         default=DEFAULT_RETRAIN,         type=str2bool, help='Retrain existing model (default: False).')
    parser.add_argument('--trained_folder',  default=DEFAULT_TRAINED_FOLDER,  type=str,      help='Floder inside output_folder containing model to retrain (default: name)')
    parser.add_argument('--flight_mode',     default=DEFAULT_FLIGHT_MODE,     type=int,      help='Flight mode (0=default).')
    parser.add_argument('--output_folder',   default=DEFAULT_OUTPUT_FOLDER,   type=str,      help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--num_agents',      default=DEFAULT_NUM_AGENTS,      type=int,      help=f'Number of agents in environment (default: {DEFAULT_NUM_AGENTS})')
    parser.add_argument('--total_timesteps', default=DEFAULT_TOTAL_TIMESTEPS, type=int,      help=f'Number of iterations to train agents over (default: {DEFAULT_TOTAL_TIMESTEPS})')
    parser.add_argument('--update_interval', default=DEFAULT_UPDATE_INTERVAL, type=int,      help=f'Intervals for training breaks (default: {DEFAULT_UPDATE_INTERVAL})')
    parser.add_argument('--n_envs',          default=DEFAULT_NUM_ENVS,        type=int,      help=f'Number of environments in vectorized training (default: {DEFAULT_NUM_ENVS})')
    ARGS = parser.parse_args()

    train(**vars(ARGS))
    print("[INFO] Done.")

    

import os
import wandb
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback

from PyFlyt.pz_envs import MAFixedwingDogfightEnvV2
from PyFlyt.pz_envs.quadx_envs.ma_combat_env import CombatWaypointPursuitEnv
from selfplay import SelfPlayEnvWings 
from wandb.integration.sb3 import WandbCallback
os.environ["WANDB_MODE"] = "disabled"  
class RandomPolicy:
    """A dummy policy that returns random actions from a given action_space."""
    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, obs, deterministic=True):
        # SB3 expects a tuple (action, state)
        return self.action_space.sample(), None


if __name__ == "__main__":

    # 1) Initialize W&B (metrics only)
    wandb.init(
        project="combat_pursuit",
        name="sac_selfplay_fixed_wing",
        config={
            "total_timesteps": int(1e7),
            "update_interval": 1000,
            "checkpoint_freq": 250_000
        },
    )

    # 2) Shared multi-agent env
    ma_env = CombatWaypointPursuitEnv(render_mode=None)
    ma_env.reset()

    random_opp = RandomPolicy(ma_env.action_space())

    # 4) Instantiate two SAC models, each on its own SelfPlayEnv
    env_ego = SelfPlayEnvWings(ma_env, train_agent_id=0, opp_policy=random_opp)
    model_ego = SAC(
        # policy="MlpPolicy",
        policy="MultiInputPolicy",
        env=env_ego,
        verbose=1,
        # tensorboard_log="./logs_fwing/tensorboard_ego_wing/",
    )

    env_adv = SelfPlayEnvWings(ma_env, train_agent_id=1, opp_policy=random_opp)
    model_adv = SAC(
        # policy="MlpPolicy",
        policy="MultiInputPolicy",
        env=env_adv,
        verbose=1,
        # tensorboard_log="./logs_fwing/tensorboard_adv_wing/",
    )

    # 5) Checkpoint + W&B callbacks
    checkpoint_ego = CheckpointCallback(
        save_freq=250_000,
        save_path="./logs_fwing/checkpoints_wing/ego/",
        name_prefix="ego_sac"
    )

    # Adv checkpoints → ./checkpoints/adv/
    checkpoint_adv = CheckpointCallback(
        save_freq=250_000,
        save_path="./logs_fwing/checkpoints_wing/adv/",
        name_prefix="adv_sac"
    )
    wandb_cb = WandbCallback(verbose=2, model_save_path=None)

    # 6) Self-play loop
    total_timesteps = int(1e7)
    update_interval = 1000
    n_iters = total_timesteps // update_interval

    for it in range(1, n_iters + 1):
        # --- Train Ego vs. fixed adversary ---
        print(f"[Iter {it}/{n_iters}]  Training Ego")
        model_ego.learn(
            total_timesteps=update_interval,
            reset_num_timesteps=False,
            callback=[checkpoint_ego, wandb_cb],
        )
        # Swap in the newly trained ego policy for the adversary’s next turn
        env_adv.opp_policy = model_ego
        ma_env.render_trajectory(f"./logs_fwing/trajectories_wing/ego_{it:04d}.png")

        # --- Train Adversary vs. fixed ego ---
        print(f"[Iter {it}/{n_iters}]  Training Adversary")
        model_adv.learn(
            total_timesteps=update_interval,
            reset_num_timesteps=False,
            callback=[checkpoint_adv, wandb_cb],
        )
        # model_ego.save(f"./final_wing_models/ego_sac_final_{it}")
        # model_adv.save(f"./final_wing_models/adv_sac_final_{it}")
        # Now fix adv in ego’s next turn
        env_ego.opp_policy = model_adv
        ma_env.render_trajectory(f"./logs_fwing/trajectories_wing/adv_{it:04d}.png")

    # 7) Final save & cleanup
    model_ego.save("./logs_fwing/final_wing_models/ego_sac_final")
    model_adv.save("./logs_fwing/final_wing_models/adv_sac_final")
    wandb.finish()

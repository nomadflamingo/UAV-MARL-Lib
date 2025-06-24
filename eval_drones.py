import os
import numpy as np
import imageio.v2 as imageio
from stable_baselines3 import SAC
from PyFlyt.pz_envs import CombatWaypointPursuitEnv

# Paths to your saved models
# EGO_MODEL_PATH = "./checkpoints_f3/ego/ego_sac_1500000_steps.zip"
# ADV_MODEL_PATH = "./checkpoints_f3/adv/adv_sac_1500000_steps.zip"


EGO_MODEL_PATH = "./checkpoints/ego/ego_sac_3750000_steps.zip"
ADV_MODEL_PATH = "./checkpoints/adv/adv_sac_3750000_steps.zip"

def make_env(render_mode: str = "human"):
    return CombatWaypointPursuitEnv(render_mode=render_mode)

def load_models():
    model_ego = SAC.load(EGO_MODEL_PATH)
    model_adv = SAC.load(ADV_MODEL_PATH)
    return model_ego, model_adv

def evaluate_and_record(ma_env, model_ego, model_adv,
                        num_episodes: int = 10,
                        output_dir: str = "videos_fwing",
                        fps: int = 30):
    os.makedirs(output_dir, exist_ok=True)
    count = 0
    for ep in range(1, num_episodes + 1):
        obs_dict, _ = ma_env.reset()
        name0, name1 = ma_env.agents

        obs_ego = obs_dict[name0]
        obs_adv = obs_dict[name1]

        total_ego = total_adv = 0.0
        done_ego = done_adv = False
        frames = []

        # grab initial camera image
        frames.append(ma_env.unwrapped.aviary.drones[0].rgbaImg)

        while not (done_ego or done_adv):
            act_ego, _ = model_ego.predict(obs_ego, deterministic=True)
            act_adv, _ = model_adv.predict(obs_adv, deterministic=True)
            # act_adv = ma_env.action_space(name1).sample()  # Random action for adversary
            actions = {name0: act_ego, name1: act_adv}
            print(f"Episode {ep:2d} → Ego action: {act_ego}, Adv action: {act_adv}")

            obs_dict, rewards, terms, truncs, _ = ma_env.step(actions)
            count += 1
            obs_ego = obs_dict[name0]
            obs_adv = obs_dict[name1]
            total_ego += rewards[name0]
            total_adv += rewards[name1]
            done_ego = terms[name0] or truncs[name0]
            done_adv = terms[name1] or truncs[name1]

            # grab the new frame
            frames.append(ma_env.unwrapped.aviary.drones[0].rgbaImg)

            if count > 250:
                break

        # write out the MP4
        path = os.path.join(output_dir, f"episode_{ep:02d}.mp4")
        writer = imageio.get_writer(path, fps=fps, codec='libx264', quality=8)
        for f in frames:
            writer.append_data(f)
        writer.close()

        print(f"Episode {ep:2d} → Ego {total_ego:.2f}, Adv {total_adv:.2f} → saved to {path}")

if __name__ == "__main__":
    # 1) Create env
    env = make_env(render_mode="human")
    # 2) Load
    ego, adv = load_models()
    # 3) Eval + record
    evaluate_and_record(env, ego, adv, num_episodes=1, output_dir="videos_drone", fps=30)

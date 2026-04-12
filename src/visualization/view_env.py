"""Quick visual test — spawns a multi-agent env and steps with random actions."""

import time
import argparse
from PyFlyt.pz_envs import MAFixedwingDogfightEnvV2
from PyFlyt.pz_envs.quadx_envs.ma_quadx_hover_env import MAQuadXHoverEnv
from PyFlyt.pz_envs.quadx_envs.ma_quadx_pursuit_evasion_env import MAQuadXPursuitEvasionEnv

ENV_REGISTRY = {
    "dogfight_FW": MAFixedwingDogfightEnvV2,
    "hover": MAQuadXHoverEnv,
    "pursuit_evasion": MAQuadXPursuitEvasionEnv,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="dogfight_FW", choices=ENV_REGISTRY.keys())
    parser.add_argument("--episodes", default=3, type=int)
    args = parser.parse_args()

    env = ENV_REGISTRY[args.env](render_mode="human")

    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            obs, rewards, terms, truncs, infos = env.step(actions)
            done = all(terms[a] or truncs[a] for a in env.agents)
            time.sleep(1.0 / 40)
        print(f"Episode {ep + 1} done")

    env.close()

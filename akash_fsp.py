import argparse
import numpy as np
import torch
import os
import copy
import wandb
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from agent.DiPo import DiPo
from agent.replay_memory import ReplayMemory, DiffusionMemory
# os.environ["WANDB_MODE"] = "disabled"
from moviepy.editor import ImageSequenceClip

# import highway_env  # noqa: F401
# from pettingzoo.mpe import simple_adversary_v3, simple_tag_v3
import vmas
import imageio
import torch.nn.functional as F

def readParser():
    parser = argparse.ArgumentParser(description='Diffusion Policy')
    parser.add_argument('--env_name',                              default="simple_tag",              help='Mujoco Gym environment (default: Hopper-v3)')
    parser.add_argument('--seed',                      type=int,   default=0,            metavar='N', help='random seed (default: 0)')
    parser.add_argument('--num_steps',                 type=int,   default=1000,         metavar='N', help='env timesteps (default: 1000000)')
    parser.add_argument('--batch_size',                type=int,   default=1024,         metavar='N', help='batch size (default: 256)')
    parser.add_argument('--gamma',                     type=float, default=0.99,         metavar='G', help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau',                       type=float, default=0.005,        metavar='G', help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--update_actor_target_every', type=int,   default=1,            metavar='N', help='update actor target per iteration (default: 1)')
    parser.add_argument("--policy_type",               type=str,   default="Diffusion",  metavar='S', help="Diffusion, VAE or MLP")
    parser.add_argument("--beta_schedule",             type=str,   default="cosine",     metavar='S', help="linear, cosine or vp")
    parser.add_argument('--n_timesteps',               type=int,   default=10,           metavar='N', help='diffusion timesteps (default: 100)')
    parser.add_argument('--diffusion_lr',              type=float, default=0.0003,       metavar='G', help='diffusion learning rate (default: 0.0003)')
    parser.add_argument('--critic_lr',                 type=float, default=0.0003,       metavar='G', help='critic learning rate (default: 0.0003)')
    parser.add_argument('--action_lr',                 type=float, default=0.03,         metavar='G', help='diffusion learning rate (default: 0.03)')
    parser.add_argument('--noise_ratio',               type=float, default=1.0,          metavar='G', help='noise ratio in sample process (default: 1.0)')
    parser.add_argument('--action_gradient_steps',     type=int,   default=10,           metavar='N', help='action gradient steps (default: 20)')
    parser.add_argument('--ratio',                     type=float, default=0.1,          metavar='G', help='the ratio of action grad norm to action_dim (default: 0.1)')
    parser.add_argument('--ac_grad_norm',              type=float, default=2.0,          metavar='G', help='actor and critic grad norm (default: 1.0)')
    parser.add_argument('--cuda',                                  default='cuda:0',                  help='run on CUDA (default: cuda:0)')
    return parser.parse_args()


def pad_obs(obs):
    """Helper function to pad observation if needed."""
    if obs.shape[-1] == 8:
        # return np.pad(obs, (0, 2), mode='constant')
        return F.pad(obs, (0, 2), mode='constant', value=0)
    return obs


class FSPTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.cuda)
        self.env = self._initialize_environment()
        self.logger = self._initialize_logger()
        agent1, agent2 = self._initialize_agents()
        self.agent1 = agent1
        self.agent2 = agent2
        self.policy_avg_ego = [copy.deepcopy(self.agent1.actor)]
        self.policy_avg_opp = [copy.deepcopy(self.agent2.actor)]
        self.p_sample_dist_ego = [1.0]
        self.p_sample_dist_opp = [1.0]
        self.global_step = 0

    def _initialize_environment(self):
        # Note: using the parallel API here.
        # return simple_adversary_v3.parallel_env(
        #     N=1, 
        #     continuous_actions=True, 
        #     render_mode="rgb_array", 
        #     max_cycles=150, 
        #     dynamic_rescaling=True
        # )
        env = vmas.make_env(
            scenario="simple_adversary", # can be scenario name or BaseScenario class
            num_envs=12,
            device="cuda", # Or "cuda" for GPU
            continuous_actions=True,
            wrapper=None,  # One of: None, "rllib", "gym", "gymnasium", "gymnasium_vec"
            max_steps=1000, # Defines the horizon. None is infinite horizon.
            seed=None, # Seed of the environment
            dict_spaces=True, # By default tuple spaces are used with each element in the tuple being an agent.
            # If dict_spaces=True, the spaces will become Dict with each key being the agent's name
            grad_enabled=False, # If grad_enabled the simulator is differentiable and gradients can flow from output to input
            terminated_truncated=True, # If terminated_truncated the simulator will return separate `terminated` and `truncated` flags in the `done()`, `step()`, and `get_from_scenario()` functions instead of a single `done` flag
        )
        return env
        # # Alternatively, you could use simple_tag_v3.parallel_env(...) if desired.

    def _initialize_logger(self):
        # dir = "record"
        # log_dir = os.path.join(dir, f'{self.args.env_name}', f'policy_type={self.args.policy_type}', f'ratio={self.args.ratio}', f'seed={self.args.seed}')
        return wandb.init(
            project="ral_mpe_adv",
            config=self.args.__dict__,
            name="dipo",
            notes="our_rewards",
        )

    def _initialize_agents(self):
        # Using the unwrapped environment to get observation and action spaces
        obs_space = self.env.observation_space['agent_0']
        act_space = self.env.action_space['agent_0']
        state_size = int(np.prod(obs_space.shape))
        action_size = int(np.prod(act_space.shape))
        memory_size = 1e6

        memory1 = ReplayMemory(state_size, action_size, memory_size, self.device)
        diffusion_memory1 = DiffusionMemory(state_size, action_size, memory_size, self.device)

        memory2 = ReplayMemory(state_size, action_size, memory_size, self.device)
        diffusion_memory2 = DiffusionMemory(state_size, action_size, memory_size, self.device)

        agent1 = DiPo(self.args, state_size, act_space, memory1, diffusion_memory1, self.device)
        agent2 = DiPo(self.args, state_size, act_space, memory2, diffusion_memory2, self.device)

        return agent1, agent2

    def update_policy_distribution(self, policy_avg, p_sample_dist):
        num_policies = len(policy_avg)
        current_time = num_policies

        avg_policy_prob = (current_time - 1) / (current_time + 1)
        latest_policy_prob = 2 / (current_time + 1)

        scaled_latest_prob = (1 / avg_policy_prob) * latest_policy_prob
        new_sample_dist = p_sample_dist + [scaled_latest_prob]

        total_sum = sum(new_sample_dist)
        normalized_sample_dist = [p / total_sum for p in new_sample_dist]

        return normalized_sample_dist

    def rescale_action(self, action):
        return (action + 1) / 2

    def compute_exploitability(self):
        """
        Evaluate exploitability using the parallel API.
        For each player (ego and opponent) we run two evaluations:
         - One using the current (average) policy.
         - One using the best response.
        The difference gives a measure of exploitability.
        """
        exp_opp_list = []
        exp_ego_list = []

        # For each player, evaluate current and best-response returns
        for player in ['ego', 'opp']:
            # ---- Evaluate current policy ----
            observations = self.env.reset(seed=100)
            episode_reward_current = 0.0
            max_steps = 300  # Define maximum steps per episode
            # Loop until episode termination (agents are removed once done)
            actions = {}
            for _ in range(max_steps):
                # for i, agent_ in enumerate(self.env.agents):
                for agent, obs in observations.items():
                    obs = pad_obs(obs)
                    # Select action based on current (average) policies.
                    if agent in ('agent_0', 'agent_1'):
                        if player == 'ego':
                            chosen_policy = np.random.choice(self.policy_avg_ego, p=self.p_sample_dist_ego)
                        else:
                            chosen_policy = np.random.choice(self.policy_avg_opp, p=self.p_sample_dist_opp)
                    elif agent in ('adversary_0', 'adversary_1'):
                        if player == 'opp':
                            chosen_policy = np.random.choice(self.policy_avg_opp, p=self.p_sample_dist_opp)
                        else:
                            chosen_policy = np.random.choice(self.policy_avg_ego, p=self.p_sample_dist_ego)
                    else:
                        chosen_policy = None

                    if chosen_policy is not None:
                        action = chosen_policy(obs, eval=True)
                        action = action.clip(-1, 1)  # Ensure action is within bounds
                    else:
                        action = None
                    actions[agent] = action if action is not None else None

                observations, rewards, terminations, truncations, infos = self.env.step(actions)
                # Accumulate rewards for the appropriate player.
                for agent, reward in rewards.items():
                    if player == 'ego' and agent in ('agent_0', 'agent_1'):
                        episode_reward_current += reward
                    elif player == 'opp' and agent in ('adversary_0', 'adversary_1'):
                        episode_reward_current += reward

            # ---- Evaluate best response ----
            observations = self.env.reset(seed=100)
            episode_reward_best_response = 0.0
            actions = {}
            # for i, agent_ in enumerate(self.env.agents):
            for _ in range(max_steps):
                for agent, obs in observations.items():
                    obs = pad_obs(obs)
                    if agent in ('agent_0', 'agent_1'):
                        if player == 'ego':  # Ego best response
                            action = self.agent1.sample_action(obs, eval=True)
                        else:
                            chosen_policy = np.random.choice(self.policy_avg_opp, p=self.p_sample_dist_opp)
                            action = chosen_policy(obs, eval=True)
                            action = action.clip(-1, 1)  # Ensure action is within bounds
                    elif agent in ('adversary_0', 'adversary_1'):
                        if player == 'opp':  # Opponent best response
                            action = self.agent2.sample_action(obs, eval=True)
                        else:
                            chosen_policy = np.random.choice(self.policy_avg_ego, p=self.p_sample_dist_ego)
                            action = chosen_policy(obs, eval=True)
                            action = action.clip(-1, 1)  # Ensure action is within bounds
                    else:
                        action = None
                    actions[agent] = action if action is not None else None

                observations, rewards, terminations, truncations, infos = self.env.step(actions)
                for agent, reward in rewards.items():
                    if player == 'ego' and agent in ('agent_0', 'agent_1'):
                        episode_reward_best_response += reward
                    elif player == 'opp' and agent in ('adversary_0', 'adversary_1'):
                        episode_reward_best_response += reward

            print(f"Episode reward current_{player}: {episode_reward_current}, Episode reward best response_{player}: {episode_reward_best_response}")
            if player == 'ego':
                exp_opp = episode_reward_best_response - episode_reward_current
                exp_opp_list.append(exp_opp)
            else:
                exp_ego = episode_reward_current - episode_reward_best_response
                exp_ego_list.append(exp_ego)

        exp_opp = torch.cat(exp_opp_list).mean() if exp_opp_list else 0.0
        exp_ego = torch.cat(exp_ego_list).mean() if exp_ego_list else 0.0
        total_exploitability = exp_opp + exp_ego
        sum_exploitability = torch.cat(exp_opp_list + exp_ego_list).sum() if exp_opp_list is not None and exp_ego_list is not None else 0.0
        print(f"Opponent exploitability: {exp_opp:.4f}, Ego exploitability: {exp_ego:.4f}")
        self.logger.log({
            "exploitability/opp": exp_opp.item(),
            "exploitability/ego": exp_ego.item(),
            "exploitability/total": total_exploitability.item(),
            "exploitability/sum": sum_exploitability.item(),
        })
        return exp_opp, exp_ego, total_exploitability, sum_exploitability

    def train_agent_against_average(self, agent_name):
        episode = 0
        steps = 0
        observations = self.env.reset()
        while steps <= self.args.num_steps:
            steps_in_env = 0
            episode_reward = 0.0

            actions = {}
            for i, agent_ in enumerate(self.env.agents):
                for agent, obs in observations.items():
                    obs = pad_obs(obs)
                    if agent_name == 'ego' and agent in ('agent_0', 'agent_1'):
                        action = self.agent1.sample_action(obs, eval=False)
                    elif agent_name == 'opp' and agent in ('adversary_0', 'adversary_1'):
                        action = self.agent2.sample_action(obs, eval=False)
                    else:
                        if agent in ('agent_0', 'agent_1'):
                            opponent_policy = np.random.choice(self.policy_avg_opp, p=self.p_sample_dist_opp)
                        elif agent in ('adversary_0', 'adversary_1'):
                            opponent_policy = np.random.choice(self.policy_avg_ego, p=self.p_sample_dist_ego)
                        action = opponent_policy(obs, eval=False)
                        action = action.clip(-1, 1)
                    actions[agent] = action if action is not None else None

                new_obs, rewards, terminations, truncations, infos = self.env.step(actions)

                done = terminations | truncations  # shape: (n_envs,)
                mask = (~done).float() * self.args.gamma  # shape: (n_envs,)

                # Store transitions and accumulate rewards
                for agent in actions.keys():
                    obs      = pad_obs(observations[agent]).detach().cpu().numpy()   # (n_envs, obs_dim)
                    next_obs = pad_obs(new_obs[agent]).detach().cpu().numpy()        # (n_envs, obs_dim)
                    act      = actions[agent].detach().cpu().numpy()                 # (n_envs, act_dim) or (n_envs,)
                    rew      = rewards[agent].detach().cpu().numpy()                 # (n_envs,)
                    mask_np  = mask.detach().cpu().numpy()                           # (n_envs,)

                    if agent in ('agent_0', 'agent_1') and agent_name == 'ego':
                        self.agent1.append_memory_batch(obs, act, rew, next_obs, mask_np)
                        episode_reward += rew.sum()  # vectorized reward accumulation
                    elif agent in ('adversary_0', 'adversary_1') and agent_name == 'opp':
                        self.agent2.append_memory_batch(obs, act, rew, next_obs, mask_np)
                        episode_reward += rew.sum()

                # Train the appropriate agent
                if agent_name == 'ego':
                    self.agent1.train(1, batch_size=self.args.batch_size, log_writer=self.logger)
                elif agent_name == 'opp':
                    self.agent2.train(2, batch_size=self.args.batch_size, log_writer=self.logger)

                steps += 1  # One joint step across all agents
                self.global_step += 1
                steps_in_env += 1
                observations = new_obs

            episode += 1
            if episode % 10 == 0:
                print(f"Episode: {episode}, steps in env: {steps_in_env}, global step: {self.global_step}")

        # Update average policy distribution
        if agent_name == 'ego':
            new_policy = copy.deepcopy(self.agent1.actor)
            self.policy_avg_ego.append(new_policy)
            self.p_sample_dist_ego = self.update_policy_distribution(self.policy_avg_ego, self.p_sample_dist_ego)
        else:
            new_policy = copy.deepcopy(self.agent2.actor)
            self.policy_avg_opp.append(new_policy)
            self.p_sample_dist_opp = self.update_policy_distribution(self.policy_avg_opp, self.p_sample_dist_opp)

    def evaluate(self, iteration):
        """Evaluate the agents using the parallel API and record a video."""
        episodes = 1
        # eval_env = self._initialize_environment()
        eval_env = vmas.make_env(
            scenario="simple_adversary", # can be scenario name or BaseScenario class
            num_envs=1,
            device="cuda", # Or "cuda" for GPU
            continuous_actions=True,
            wrapper=None,  # One of: None, "rllib", "gym", "gymnasium", "gymnasium_vec"
            max_steps=300, # Defines the horizon. None is infinite horizon.
            seed=None, # Seed of the environment
            dict_spaces=True, # By default tuple spaces are used with each element in the tuple being an agent.
            # If dict_spaces=True, the spaces will become Dict with each key being the agent's name
            grad_enabled=False, # If grad_enabled the simulator is differentiable and gradients can flow from output to input
            terminated_truncated=True, # If terminated_truncated the simulator will return separate `terminated` and `truncated` flags in the `done()`, `step()`, and `get_from_scenario()` functions instead of a single `done` flag
        )
        
        video_folder = "dipo_vmas_videos"
        os.makedirs(video_folder, exist_ok=True)
        returns = torch.zeros((episodes,), dtype=torch.float32)

        for gg in range(episodes):
            observations = eval_env.reset(seed=100 + gg)
            episode_reward = 0.0
            video_file = os.path.join(video_folder, f"episode_{iteration}.mp4")
            actions = {}
            frame_list = []
            max_steps = 300
            for _ in range(max_steps):
                for i, agents_ in enumerate(eval_env.agents):
                    for agent, obs in observations.items():
                        obs = pad_obs(obs)
                        if agent in ('agent_0', 'agent_1'):
                            action = self.agent1.sample_action(obs, eval=True)
                        elif agent in ('adversary_0', 'adversary_1'):
                            action = self.agent2.sample_action(obs, eval=True)
                        else:
                            raise ValueError(f"Unexpected agent: {agent}")
                        actions[agent] = action if action is not None else None

                    observations, rewards, terminations, truncations, infos = eval_env.step(actions)
                    # Sum rewards across agents.
                    # observations = new_obs
                    for reward in rewards.values():
                        episode_reward += reward

                    # Render and record the frame.
                    frame = eval_env.render(
                        mode="rgb_array",
                        agent_index_focus=None,  # Can give the camera an agent index to focus on
                    )
                    frame_list.append(frame)
                    
                returns[gg] = episode_reward


        mean_return = torch.mean(returns)
        self.logger.log({
            "reward/test": mean_return.item(),
            "steps": self.global_step,
        })
        fps=10
        clip = ImageSequenceClip(frame_list, fps=fps)
        # clip.write_gif(filename = video_file, fps=fps)
        clip.write_videofile(video_file, fps=fps)


        print('-' * 60)
        print(f'Num steps: {self.global_step:<5}  reward: {mean_return:<5.1f}')
        print('-' * 60)
        # eval_env.close()

    def populate_buffer(self, seed=42, gamma=0.9, buffer_size=10000):
        """Populate replay buffer using the parallel API with vectorized envs."""
        steps = 0
        observations = self.env.reset(seed=seed)  # set seed once if needed

        while steps <= buffer_size:
            actions = {}
            for agent_ in self.env.agents:
                # Vectorized action sampling across envs for each agent
                for agent, obs in observations.items():
                    obs = pad_obs(observations[agent])  # shape: (n_envs, obs_dim)
                    actions[agent] = self.env.get_random_action(agent_)

            # Step the environment
            new_obs, rewards, terminations, truncations, infos = self.env.step(actions)

            # Compute done mask (shared across all agents per env)
            done = terminations | truncations  # shape: (n_envs,)
            mask = (~done).float() * gamma     # shape: (n_envs,)

            for agent in actions:
                obs      = pad_obs(observations[agent]).view(len(done), -1)   # (n_envs, obs_dim)
                next_obs = pad_obs(new_obs[agent]).view(len(done), -1)
                act      = actions[agent]   # (n_envs, act_dim) or (n_envs,)
                rew      = rewards[agent]   # (n_envs,)

                if agent in ('agent_0', 'agent_1'):
                    self.agent1.append_memory_batch(
                        obs.detach().cpu().numpy(),
                        act.detach().cpu().numpy(),
                        rew.detach().cpu().numpy(),
                        next_obs.detach().cpu().numpy(),
                        mask.detach().cpu().numpy()
                    )
                elif agent in ('adversary_0', 'adversary_1'):
                    self.agent2.append_memory_batch(
                        obs.detach().cpu().numpy(),
                        act.detach().cpu().numpy(),
                        rew.detach().cpu().numpy(),
                        next_obs.detach().cpu().numpy(),
                        mask.detach().cpu().numpy()
                    )

            steps += len(done)
            if steps >= buffer_size:
                print("Buffer populated.")
                return

            observations = new_obs

    def main(self):
        print("Populating buffer...")
        self.populate_buffer()
        os.makedirs('vmas_adv_ckpts/dipo_large_mpe_save_model_1', exist_ok=True)
        os.makedirs('vmas_adv_ckpts/dipo_large_mpe_save_model_2', exist_ok=True)

        fsp_iterations = 1000
        for fsp_iter in range(fsp_iterations):
            for agent_name in ['opp', 'ego']:
                self.train_agent_against_average(agent_name)

            exp_opp, exp_ego, exploitability, sum_exploitability = self.compute_exploitability()
            self.logger.log({
                "fsp_iteration": fsp_iter + 1,
                "exploitability/total": exploitability,
                "exploitability/mean": sum_exploitability,
                "exploitability/opp": exp_opp,
                "exploitability/ego": exp_ego,
            })

            print(f"Exploitability after FSP Iteration {fsp_iter + 1}: {exploitability:.4f}")
            print(f"Mean Exploitability: {sum_exploitability:.4f}")
            if fsp_iter % 50 == 0: #and fsp_iter > 0:
                self.evaluate(fsp_iter)
            if fsp_iter % 100 == 0: #and fsp_iter > 0:
                self.agent1.save_model(dir='mpe_adv_ckpts/dipo_large_mpe_save_model_1', id=fsp_iter)
                self.agent2.save_model(dir='mpe_adv_ckpts/dipo_large_mpe_save_model_2', id=fsp_iter)

        self.evaluate(fsp_iter)
        self.agent1.save_model(dir='mpe_adv_ckpts/dipo_large_mpe_save_model_1', id=fsp_iter)
        self.agent2.save_model(dir='mpe_adv_ckpts/dipo_large_mpe_save_model_2', id=fsp_iter)
        print("FSP training complete.")


if __name__ == "__main__":
    args = readParser()
    trainer = FSPTrainer(args)
    trainer.main()

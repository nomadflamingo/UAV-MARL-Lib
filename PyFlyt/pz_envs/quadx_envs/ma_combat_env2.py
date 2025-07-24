from __future__ import annotations

from typing import Literal, Any
import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import time

from PyFlyt.pz_envs.quadx_envs.ma_quadx_base_env import MAQuadXBaseEnv
from PyFlyt.gym_envs.utils.waypoint_handler import WaypointHandler

class CombatWaypointPursuitEnv2(MAQuadXBaseEnv):
    """
    Multi-Agent Combat Waypoint Pursuit Environment.

    Ego agent pursues waypoints while the adversary tries to catch the ego.
    Agents receive competitive and agility-based rewards, and can observe waypoint deltas.
    """
    metadata = {"render_modes": ["human"], "name": "combat_pursuit"}

    def __init__(
        self,
        ego_start_pos: np.ndarray = np.array([[0.0, 0.0, 1.0]]),
        adv_start_pos: np.ndarray = np.array([[2.0, 2.0, 1.0]]),
        max_lin_vel: float        = 5.0,    # expected max linear speed
        time_penalty: float       = 0.01,   # per-step penalty
        shaping_coeff: float      = 1.0,    # for ego potential shaping
        closing_coeff: float      = 1.0,    # for adv closing bonus
        waypoint_reward: float    = 10.0,   # bonus on reaching a waypoint
        catch_reward: float       = 10.0,   # bonus when adv catches ego
        sparse_reward: bool = False,
        num_targets: int = 1,
        use_yaw_targets: bool = False,
        goal_reach_distance: float = 0.2,
        goal_reach_angle: float = 0.1,
        flight_mode: int = 0,
        flight_dome_size: float = 5.0,
        max_duration_seconds: float = 10.0,
        angle_representation: Literal["euler", "quaternion"] = "quaternion",
        agent_hz: int = 30,
        render_mode: None | str = None,
    ):
        self.max_lin_vel      = max_lin_vel
        self.time_penalty     = time_penalty
        self.shaping_coeff    = shaping_coeff
        self.closing_coeff    = closing_coeff
        self.waypoint_reward  = waypoint_reward
        self.catch_reward     = catch_reward

        self.n_ego = ego_start_pos.shape[0]
        self.n_adv = adv_start_pos.shape[0]
        self.n_agents = self.n_ego + self.n_adv

        # Indices for ego and adversary
        self.ego_indices = list(range(self.n_ego))
        self.adv_indices = list(range(self.n_ego, self.n_agents))

        # Save target config for padding
        self.num_targets = num_targets
        self.target_dim = 4 if use_yaw_targets else 3
        
        self.agent_hz = agent_hz

        # Setup starting positions and orientations
        start_pos = np.vstack([ego_start_pos, adv_start_pos])
        start_orn = np.zeros_like(start_pos)
        super().__init__(
            start_pos=start_pos,
            start_orn=start_orn,
            flight_mode=flight_mode,
            flight_dome_size=flight_dome_size,
            max_duration_seconds=max_duration_seconds,
            angle_representation=angle_representation,
            agent_hz=agent_hz,
            render_mode=render_mode,
        )
        self.sparse_reward = sparse_reward

        # Waypoint handler for multi-target pursuit
        self.waypoints = WaypointHandler(
            enable_render=self.render_mode is not None,
            num_targets=num_targets,
            use_yaw_targets=use_yaw_targets,
            goal_reach_distance=goal_reach_distance,
            goal_reach_angle=goal_reach_angle,
            flight_dome_size=flight_dome_size,
            min_height=0.1,
            np_random=np.random.default_rng(),
        )

        # Agent-specific observation space (attitude + fixed-shape target deltas)
        target_dim = 4 if use_yaw_targets else 3
        self._agent_observation_space = spaces.Dict({
            "attitude": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.combined_space.shape[0] + 3,),
                dtype=np.float64,
            ),
            "target_deltas": spaces.Box(
                low=-2 * flight_dome_size,
                high=2 * flight_dome_size,
                shape=(num_targets, target_dim),
                dtype=np.float64,
            ),
        })

        self.max_reward_inc = 10                                # Maximum reward per step
        self.max_reward = self.max_reward_inc * self.max_steps  # Maximum reward per episode

        # Trajectory logs
        self.ego_traj = {i: [] for i in self.ego_indices}
        self.adv_traj = {i: [] for i in self.adv_indices}

    def observation_space(self, agent: Any = None) -> spaces.Space:
        """Return the per-agent observation space."""
        return self._agent_observation_space

    def action_space(self, agent: Any = None) -> spaces.Space:
        """Forward action space to the base environment."""
        return super().action_space(agent)

    def reset(
        self, *, seed: None | int = None, options: None | dict[str, Any] = None
    ):
        super().begin_reset(seed, options)
        # Reset waypoints and clear trajectories
        self.waypoints.reset(self.aviary, np.random.default_rng())
        self.ego_traj.clear()
        self.adv_traj.clear()
        super().end_reset()

        self.agents = [f"ego_{i}" for i in self.ego_indices] + [f"adv_{i}" for i in self.adv_indices]
        self.agent_name_mapping = {name: i for i, name in enumerate(self.agents)}

        observations = {
            ag: self.compute_observation_by_id(self.agent_name_mapping[ag])
            for ag in self.agents
        }
        infos = {ag: {} for ag in self.agents}
        return observations, infos

    def compute_observation_by_id(self, agent_id: int) -> dict[str, np.ndarray]:
        """Compute observation for a single agent by ID and pad waypoint deltas."""
        # print(f"Agent ID: {agent_id}")
        raw = self.compute_attitude_by_id(agent_id)
        # self.attitudes = np.stack(self.aviary.all_states, axis=0)
        aux = self.aviary.aux_state(agent_id)
        ang_vel, ang_pos, lin_vel, lin_pos, quat = raw

        # Build attitude vector
        if self.angle_representation == 0:
            attitude = np.concatenate(
                [ang_vel, ang_pos, lin_vel, lin_pos,
                 self.past_actions[agent_id], aux],
                axis=-1,
            )
        else:
            attitude = np.concatenate(
                [ang_vel, quat, lin_vel, lin_pos,
                 self.past_actions[agent_id], aux],
                axis=-1,
            )

        # Compute deltas to all waypoints
        # deltas = self.waypoints.distance_to_targets(ang_pos, lin_pos, quat)
        # if deltas.shape[0] < self.num_targets:
        #     pad_len = self.num_targets - deltas.shape[0]
        #     pad = np.zeros((pad_len, self.target_dim), dtype=deltas.dtype)
        #     deltas = np.vstack([deltas, pad])
        # else:
        #     deltas = deltas[: self.num_targets]

        # ─── Safe distance_to_targets: if no targets, skip the call ─────
        if len(self.waypoints.targets) == 0:
            deltas = np.zeros((0, self.target_dim), dtype=np.float64)
        else:
            raw = np.asarray(self.waypoints.distance_to_targets(ang_pos, lin_pos, quat))
            if raw.ndim == 1:
                # single target → shape (1, target_dim)
                deltas = raw.reshape(1, -1)
            else:
                # already 2-D
                deltas = raw

        # ─── now pad/truncate as before ─────────────────────────────────
        if deltas.shape[0] < self.num_targets:
            pad_len = self.num_targets - deltas.shape[0]
            pad = np.zeros((pad_len, self.target_dim), dtype=deltas.dtype)
            deltas = np.vstack([deltas, pad])
        else:
            deltas = deltas[: self.num_targets]

        # Log current position for trajectory
        pos = lin_pos.copy()
        if agent_id in self.ego_indices:
            self.ego_traj[agent_id].append(pos)
        else:
            self.adv_traj[agent_id].append(pos)

        return {"attitude": attitude, "target_deltas": deltas}

    def compute_term_trunc_reward_info_by_id(self, agent_id: int):
        """Compute termination, truncation, reward and info for one agent."""
        raw = self.compute_attitude_by_id(agent_id)
        _, ang_pos, lin_vel, lin_pos, quat = raw

        # default flags
        trunc = self.step_count > self.max_steps
        term  = False
        info  = {}

        # 1) normalized agility
        v_norm = (np.linalg.norm(lin_vel) / self.max_lin_vel)

        # 2) compute delta to next waypoint (vector from curr_pos → wp)
        if len(self.waypoints.targets)==0:
            curr_delta = np.zeros((self.target_dim,),dtype=np.float64)
        else:
            curr_delta = np.asarray(self.waypoints.distance_to_targets(ang_pos, lin_pos, quat))#[0]
        curr_dist = float(np.linalg.norm(curr_delta[:3]))
        dist_norm = curr_dist / self.flight_dome_size

        # 3) find previous position
        if agent_id in self.ego_indices:
            prev_pos = np.array(self.ego_traj[agent_id][-1])
        else:
            prev_pos = np.array(self.adv_traj[agent_id][-1])

        # recompute target point:  wp = curr_pos + curr_delta
        target_pt = lin_pos + curr_delta[:3]
        prev_dist = np.linalg.norm(prev_pos - target_pt)

        # 4) shaping reward (ego only)
        shape_r = 0.0
        if agent_id in self.ego_indices:
            # shape_r = self.shaping_coeff * (prev_dist - curr_dist) / self.flight_dome_size
            shape_r =  - curr_dist
        # 5) time penalty
        tp = self.time_penalty

        # now build raw reward
        # some thing dumb 
        yaw = ang_pos[-1]
        dist_origin = float(np.linalg.norm(lin_pos))

        boundary_r  = np.tanh(0.1 * yaw       - 1.0)
        boundary_r -= np.tanh(0.0025 * dist_origin - 1.0)

        # reward = v_norm - dist_norm + shape_r - tp + boundary_r
        reward = shape_r + boundary_r

        # safety termination condition: crash or dome violation
        x, y, z = lin_pos
        crashed = z <= 0.0
        outside_dome = np.linalg.norm(lin_pos) > self.flight_dome_size+0.5

        if crashed or outside_dome:
            term = True
            info["crashed"] = crashed
            info["outside_dome"] = outside_dome
            reward -= 10.0  # penalty for violating safety
            reward = float(np.clip(reward, -self.max_reward_inc, self.max_reward_inc))
            reward = reward/self.max_reward
            return term, trunc, reward, info

        # bonus on event
        if agent_id == self.ego_index:
            # progress fraction
            prog = self.waypoints.progress_to_next_target
            reward += max(3.0 * prog, 0.0)
            reward += max(1.0 / self.waypoints.distance_to_next_target, 0.0)
            # waypoint reached?
            if curr_dist < self.waypoints.goal_reach_distance:
                reward += self.waypoint_reward
                info["waypoint_reached"] = True
                self.waypoints.advance_targets()
                # self.truncation |= self.waypoints.all_targets_reached
                info['num_targets_reached'] = self.waypoints.num_targets_reached
                info['env_complete'] = self.waypoints.all_targets_reached
                # self.info["env_complete"] = self.waypoints.all_targets_reached
                # self.info["num_targets_reached"] = self.waypoints.num_targets_reached
                term = True # Waypoint reached
        else:
            # # adversary closing bonus
            # # compute current ego position
            # ego_pos = self.aviary.state(self.ego_index)[-1]
            # prev_adv    = np.array(self.adv_traj[-1])
            # prev_ego    = np.array(self.ego_traj[-1])
            # # prev_dist_ae= np.linalg.norm(prev_adv - prev_ego)
            # # curr_dist_ae= np.linalg.norm(lin_pos - ego_pos)
            # # close_r     = self.closing_coeff * (prev_dist_ae - curr_dist_ae) / self.flight_dome_size
            # curr_dist_ae = float(np.linalg.norm(lin_pos - ego_pos))
            # reward     -= curr_dist_ae
            # # catch bonus?
            # if curr_dist_ae < 0.3:
            #     reward += self.catch_reward
            #     info["ego_caught"] = True
            #     term = True
            # For adversaries, reward closing distance to nearest ego
            ego_positions = [self.aviary.state(eid)[-1] for eid in self.ego_indices]
            ego_dists = [np.linalg.norm(lin_pos - ep) for ep in ego_positions]
            min_dist = min(ego_dists)
            reward -= min_dist
            if min_dist < 0.3:
                reward += self.catch_reward
                term = True

        # clip
        reward = float(np.clip(reward, -self.max_reward_inc, self.max_reward_inc))
        reward = reward/self.max_reward
        return term, trunc, reward, info
    
    def step(self, actions: dict[str, np.ndarray]) -> tuple[
        dict[str, Any],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        """step.

        Args:
            actions (dict[str, np.ndarray]): actions

        Returns:
            tuple[dict[str, Any], dict[str, float], dict[str, bool], dict[str, bool], dict[str, dict[str, Any]]]:

        """
        # copy over the past actions
        self.past_actions = self.current_actions.copy()

        # set the new actions and send to aviary
        self.current_actions *= 0.0
        for k, v in actions.items():
            self.current_actions[self.agent_name_mapping[k]] = v
        self.aviary.set_all_setpoints(self.current_actions)

        # observation and rewards dictionary
        observations = dict()
        terminations = {k: False for k in self.agents}
        truncations = {k: False for k in self.agents}
        rewards = {k: 0.0 for k in self.agents}
        infos = {k: dict() for k in self.agents}

        # step enough times for one RL step
        for _ in range(self.env_step_ratio):
            self.aviary.step()
            self.update_states()

            # update reward, term, trunc, for each agent
            # TODO: make it so this doesn't have to be computed every aviary step
            for ag in self.agents:
                ag_id = self.agent_name_mapping[ag]

                # compute term trunc reward
                term, trunc, rew, info = self.compute_term_trunc_reward_info_by_id(
                    ag_id
                )
                terminations[ag] |= term
                truncations[ag] |= trunc
                rewards[ag] += rew
                infos[ag].update(info)

                # compute observations
                observations[ag] = self.compute_observation_by_id(ag_id)

                if len(self.waypoints.targets) == 0:
                    terminations[ag] = True
        # increment step count and cull dead agents for the next round
        self.step_count += 1
        self.agents = [
            agent
            for agent in self.agents
            if not (terminations[agent] or truncations[agent])
        ]

        return observations, rewards, terminations, truncations, infos

    def render(self):
        """Render the environment and waypoints."""
        # super().render()
        # if self.render_mode:
        #     self.waypoints.render()

        # if self.render:
        #     elapsed = time.time() - self.now
        #     self.now = time.time()

        #     self._sim_elapsed += self.step_period
        #     self._frame_elapsed += elapsed

        #     time.sleep(max(self._sim_elapsed - self._frame_elapsed, 0.0))

        #     # print RTF every 0.5 seconds, this actually adds considerable overhead
        #     if self._frame_elapsed >= 0.5:
        #         # calculate real time factor based on realtime/simtime
        #         RTF = self._sim_elapsed / (self._frame_elapsed + 1e-6)
        #         self._sim_elapsed = 0.0
        #         self._frame_elapsed = 0.0

        #         self.rtf_debug_line = self.addUserDebugText(
        #             text=f"RTF: {RTF:.3f}",
        #             textPosition=[0, 0, 0],
        #             textColorRGB=[1, 0, 0],
        #             replaceItemUniqueId=self.rtf_debug_line,
        #         )
        print('please render')

    def render_trajectory(self, save_path: str = None):
        """Plot 3D trajectories for ego and adversary, and current target."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ego_np = np.array(self.ego_traj)
        adv_np = np.array(self.adv_traj)
        ax.plot(ego_np[:, 0], ego_np[:, 1], ego_np[:, 2], label="Ego")
        ax.plot(adv_np[:, 0], adv_np[:, 1], adv_np[:, 2], label="Adversary")

        if len(self.ego_traj) > 0:
            last_pos = self.ego_traj[-1]
            _, ang_pos, _, _, quat = self.compute_attitude_by_id(self.ego_index)
            delta = self.waypoints.distance_to_targets(ang_pos, last_pos, quat)
            current_delta = delta[0] if delta.shape[0] > 0 else np.zeros(self.target_dim)
            target_pt = last_pos + current_delta[:3]
            ax.scatter(*target_pt, color="red", marker="x", s=100, label="Next Target")

        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.legend(); ax.set_title("Trajectories")
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()

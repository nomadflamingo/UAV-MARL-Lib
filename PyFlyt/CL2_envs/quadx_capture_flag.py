
from typing import Any, Literal

import numpy as np
import time
from gymnasium import spaces

from PyFlyt.pz_envs.quadx_envs.ma_quadx_base_env import MAQuadXBaseEnv
from PyFlyt.CL2_envs.utils.flag_handler import FlagHandler
from PyFlyt.CL2_envs.utils.utils import generate_circle_points

class QuadXCaptureFlagEnv(MAQuadXBaseEnv):
    """QuadX Capture the Flag envrironment
    
    Actions are defined by flight_mode
    The target is a set of '[x, y, z]' waypoints in space
    
    Args:
        sparse_reward (bool): wether to use sparse rewards or not
        num_flags (int): the number of flag stations in the environment.
        flag_reach_distance (float): radius around flag considered to be reached.
        flight_mode (int): the flight mode of UAVs.
        flight_dome_size (float): size of the allowable flying area.
        max_duration_seconds (float): maximum simulation time of the environment.
        angle_representation (Literal["euler", "quaternion"]): can be "euler" or "quaternion".
        agent_hz (int): looprate of the gaent to envirionment interaction.
        render_mode (None | Literal[]"human", "rgb_array"]): render_mode
        render_resolution (tuple[int, int]): render_resolution

    """

    metadata = {
        "render_modes": ["human"],
        "name": "ma_quadx_ctf",
    }

    def __init__(self, 
                 sparse_reward: bool = False,
                 num_agents: int = 5,
                 num_flags: int = 4,
                 flag_reach_distance: float = 0.2,
                 flight_mode: int = 0, 
                 flight_dome_size: float = 5.0, 
                 max_duration_seconds: float = 60.0, 
                 angle_representation: Literal["euler", "quaternion"] = "quaternion", 
                 agent_hz: int = 30, 
                render_mode: None | Literal["human", "rgb_array"] = None,
                render_resolution: tuple[int, int] = (480, 480),
                 ):

        self.agent_hz = agent_hz
        
        # Spawn agnet positions
        self.height = 1.0
        xy_list = generate_circle_points(radius=flight_dome_size/5.0, n=num_agents)
        start_pos0 = [[float(x), float(y), self.height] for (x, y) in xy_list]
        start_pos = np.array(start_pos0, dtype=np.float32)
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
        
        # Define Flags
        self.flags = FlagHandler(
                                enable_render=self.render_mode is not None,
                                num_flags=num_flags,
                                flag_reach_distance=flag_reach_distance,
                                flag_spread_raduis=1.0,
                                flight_dome_size=flight_dome_size,
                                height=self.height
                            )
        
        # Define Teams
        self.agent_teams = {}  # map agent name to team name
        team_names = ['red', 'blue']  # you can expand this if needed

        for i, agent in enumerate(range(num_agents)):
            team = team_names[i % len(team_names)]  # alternate teams
            self.agent_teams[f"uav_{i}"] = team
        
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
                shape=(num_flags, 3),
                dtype=np.float64,
            ),
        })

    def observation_space(self, agent: Any = None) -> spaces.Space:
        """Return the per-agent observation space."""
        return self._agent_observation_space

    def action_space(self, agent: Any = None) -> spaces.Space:
        """Forward action space to the base environment."""
        return super().action_space(agent)

    def reset(
        self, *, seed: None | int = None, options: None | dict[str, Any] = None
    ):
        # seed the RNG
        # print("Seed", seed)
        np_random = np.random.RandomState(seed=seed)

        # start out pointing in outward directions equally spaced
        start_x = np_random.uniform(
            low=-1,
            high=1,
            size=(self.num_possible_agents,),
        )
        start_y = np_random.uniform(
            low=-1,
            high=1,
            size=(self.num_possible_agents,),
        )
        start_z = np_random.uniform(
            low=0.1,
            high=1.2,
            size=(self.num_possible_agents,),
        )

        # define the starting positions
        start_pos = np.zeros((self.num_possible_agents, 3))
        start_pos[:, 0] = start_x
        start_pos[:, 1] = start_y
        start_pos[:, 2] = start_z

        self.start_pos = start_pos

        super().begin_reset(seed, options)
        # Reset waypoints and clear trajectories
        self.flags.reset(self.aviary, np.random.default_rng())

        # Colour teams
        # if we're rendering, set the colors of the wingtips and tail components
        # if self.render_mode:
        #     for agent_id in range(self.num_possible_agents):
        #         # wingtips and tail component IDs
        #         for component_id in [1, 2, 3, 4]:
        #             self.aviary.changeVisualShape(
        #                 self.aviary.drones[agent_id].Id,
        #                 -1,
        #                 rgbaColor=(
        #                     np.array([1.0, 0.0, 0.0, 1.0])
        #                     if (agent_id+1)%2
        #                     else np.array([0.0, 0.0, 1.0, 1.0])
        #                 ),
        #             )

        super().end_reset()

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

        # # ─── Safe distance_to_targets: if no targets, skip the call ─────
        # if len(self.flags.targets) == 0:
        #     deltas = np.zeros((0, self.target_dim), dtype=np.float64)
        # else:
        #     raw = np.asarray(self.flags.distance_to_targets(ang_pos, lin_pos, quat))
        #     if raw.ndim == 1:
        #         # single target → shape (1, target_dim)
        #         deltas = raw.reshape(1, -1)
        #     else:
        #         # already 2-D
        #         deltas = raw

        # # ─── now pad/truncate as before ─────────────────────────────────
        # if deltas.shape[0] < self.num_targets:
        #     pad_len = self.num_targets - deltas.shape[0]
        #     pad = np.zeros((pad_len, self.target_dim), dtype=deltas.dtype)
        #     deltas = np.vstack([deltas, pad])
        # else:
        #     deltas = deltas[: self.num_targets]

        # # Log current position for trajectory
        # pos = lin_pos.copy()
        # if agent_id == self.ego_index:
        #     self.ego_traj.append(pos)
        # else:
        #     self.adv_traj.append(pos)

        deltas = np.zeros((4,3))

        return {"attitude": attitude, "target_deltas": deltas}
    
    def compute_term_trunc_reward_info_by_id(self, agent_id: int):
        """Compute termination, truncation, reward and info for one agent."""
        # raw = self.compute_attitude_by_id(agent_id)

        # default flags
        trunc = False
        term  = False
        info  = {}
        reward = 0

        return trunc, term, reward, info

    
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

            for flag_idx, flag_pos in enumerate(self.flags.targets):
                for ag in self.agents:
                    ag_id = self.agent_name_mapping[ag]
                    lin_pos = self.aviary.state(ag_id)[3]  # linear position from state


                    dist = np.linalg.norm(lin_pos - flag_pos)
                    # if dist < self.flags.flag_reach_distance:
                    #     team = self.agent_teams[ag]
                    #     color = self.flags.team_colors[team]
                    #     if self.render_mode:
                    #         self.flags.p.changeVisualShape(
                    #             self.flags.target_visual[flag_idx],
                    #             linkIndex=-1,
                    #             rgbaColor=color,
                    #         )

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

                if len(self.flags.targets) == 0:
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
        """Render the environment and flags."""
        # super().render()
        # if self.render_mode:
        #     self.flags.render()

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
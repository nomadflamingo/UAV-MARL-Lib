"""Multiagent QuadX Pursuit-Evasion Environment.

2v2 (configurable) pursuit-evasion game: pursuers try to capture evaders
within a spherical arena. Compatible with SAC + MlpPolicy via MASelfPlayEnv.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml
from gymnasium import spaces

from PyFlyt.gym_envs.utils.dome_renderer import capture_frame as _capture_frame
from PyFlyt.gym_envs.utils.dome_renderer import draw_dome
from PyFlyt.pz_envs.quadx_envs.ma_quadx_base_env import MAQuadXBaseEnv


_CONFIG_PATH = Path(__file__).parents[3] / "configs" / "pursuit_evasion.yaml"


def _load_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load YAML defaults and apply any kwarg overrides.

    env_params nested section is flattened into the top level so the env can
    read keys directly.  Caller overrides (flat kwargs) are applied last.
    """
    with open(_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    # Promote env_params keys to top level
    env_params = cfg.pop("env_params", {})
    cfg.update(env_params)
    if overrides:
        cfg.update(overrides)
    return cfg


class MAQuadXPursuitEvasionEnv(MAQuadXBaseEnv):
    """2v2 Pursuit-Evasion environment for QuadX drones.

    Pursuers (uav_0, uav_1) try to capture evaders (uav_2, uav_3)
    by getting within capture_distance. Episode ends on first capture
    or time limit.

    Actions: [vp, vq, vr, T] (angular rates + thrust), flight_mode=0.
    Observations: flat Box vector (self-state + teammate + opponents + boundary).
    """

    metadata = {
        "render_modes": ["human"],
        "name": "ma_quadx_pursuit_evasion",
        "is_parallelizable": True,
    }

    def __init__(self, render_mode: str | None = None, **kwargs):
        cfg = _load_config(kwargs)

        # --- Game parameters ---
        self.num_pursuers: int = int(cfg["num_pursuers"])
        self.num_evaders: int = int(cfg["num_evaders"])
        self.capture_distance: float = float(cfg["capture_distance"])
        self.terminate_on_first_capture: bool = bool(cfg["terminate_on_first_capture"])

        # --- Arena ---
        flight_dome_size: float = float(cfg["flight_dome_size"])
        self.boundary_penalty_fraction: float = float(cfg["boundary_penalty_fraction"])

        # --- Spawning ---
        self.spawn_min_radius: float = float(cfg["spawn_min_radius"])
        self.spawn_max_radius: float = float(cfg["spawn_max_radius"])
        self.spawn_min_height: float = float(cfg["spawn_min_height"])
        self.spawn_max_height: float = float(cfg["spawn_max_height"])

        # --- Vision ---
        self.vision_distance_pursuer: float = float(cfg["vision_distance_pursuer"])
        self.vision_distance_evader: float = float(cfg["vision_distance_evader"])

        # --- Rewards: Pursuer ---
        self.capture_reward: float = float(cfg["capture_reward"])
        self.team_capture_reward: float = float(cfg["team_capture_reward"])
        self.all_captured_reward: float = float(cfg["all_captured_reward"])
        self.distance_reward_scale: float = float(cfg["distance_reward_scale"])
        self.time_penalty: float = float(cfg["time_penalty"])
        self.boundary_penalty_scale: float = float(cfg["boundary_penalty_scale"])
        self.collision_penalty: float = float(cfg["collision_penalty"])

        # --- Rewards: Evader ---
        self.captured_penalty: float = float(cfg["captured_penalty"])
        self.teammate_captured_penalty: float = float(cfg["teammate_captured_penalty"])
        self.all_survived_reward: float = float(cfg["all_survived_reward"])
        self.survival_reward_per_step: float = float(cfg["survival_reward_per_step"])
        self.evader_distance_reward_scale: float = float(
            cfg["evader_distance_reward_scale"]
        )

        # --- Simulation ---
        flight_mode: int = int(cfg["flight_mode"])
        max_duration_seconds: float = float(cfg["max_duration_seconds"])
        angle_representation: str = str(cfg["angle_representation"])
        agent_hz: int = int(cfg["agent_hz"])

        # --- Team structure ---
        num_agents = self.num_pursuers + self.num_evaders
        self.is_evader = np.concatenate(
            [
                np.zeros(self.num_pursuers, dtype=bool),
                np.ones(self.num_evaders, dtype=bool),
            ]
        )
        self.pursuer_ids = np.where(~self.is_evader)[0]
        self.evader_ids = np.where(self.is_evader)[0]

        # Placeholder start positions (overwritten in reset)
        start_pos = np.zeros((num_agents, 3))
        start_orn = np.zeros((num_agents, 3))

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

        # --- Observation space ---
        # Per agent: self(20) + teammate(6) + opponents(7 each) + boundary(1)
        num_teammates = max(self.num_pursuers, self.num_evaders) - 1
        num_opponents = max(self.num_pursuers, self.num_evaders)
        self.obs_size = (
            self.combined_space.shape[0]  # 20 (euler) or 21 (quaternion)
            + num_teammates * 6  # relative pos + vel per teammate
            + num_opponents * 7  # vision_flag + relative pos + vel per opponent
            + 1  # boundary distance
        )
        self._observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_size,), dtype=np.float64
        )

        # --- Precomputed team lookups (static across episode) ---
        self._teammate_ids = {}
        self._opponent_ids = {}
        for aid in range(num_agents):
            if self.is_evader[aid]:
                self._teammate_ids[aid] = [i for i in self.evader_ids if i != aid]
                self._opponent_ids[aid] = self.pursuer_ids
            else:
                self._teammate_ids[aid] = [i for i in self.pursuer_ids if i != aid]
                self._opponent_ids[aid] = self.evader_ids
        self._pursuer_id_to_idx = {
            int(pid): idx for idx, pid in enumerate(self.pursuer_ids)
        }

        # --- Runtime state ---
        self.captured = np.zeros(num_agents, dtype=bool)
        self.pairwise_distances = np.zeros((num_agents, num_agents))
        self.prev_min_dist_to_evader = np.full(self.num_pursuers, np.inf)
        self.capture_happened = False
        self._np_random = np.random.default_rng()
        self._dome_line_ids: list[int] = []
        self._cached_info: dict[str, Any] = {}
        self._frame_width: int = 720
        self._frame_height: int = 720

    def observation_space(self, agent: Any = None) -> spaces.Box:
        """Return flat Box observation space (same for all agents)."""
        return self._observation_space

    def _get_start_pos_orn(
        self, seed: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate random spawn positions with team separation.

        Pursuers spawn in one semicircle (angles 0..pi),
        evaders in the opposite semicircle (angles pi..2*pi).
        """
        rng = np.random.default_rng(seed)

        start_pos = np.zeros((self.num_possible_agents, 3))
        start_orn = np.zeros((self.num_possible_agents, 3))

        angle_offset = rng.uniform(0.0, 2 * np.pi)

        for i in range(self.num_possible_agents):
            if i < self.num_pursuers:
                # Pursuer: angles in [0, pi)
                angle = rng.uniform(0.0, np.pi) + angle_offset
            else:
                # Evader: angles in [pi, 2*pi)
                angle = rng.uniform(np.pi, 2 * np.pi) + angle_offset

            radius = rng.uniform(self.spawn_min_radius, self.spawn_max_radius)
            height = rng.uniform(self.spawn_min_height, self.spawn_max_height)

            start_pos[i] = [
                radius * np.cos(angle),
                radius * np.sin(angle),
                height,
            ]
            # Face toward the arena center
            start_orn[i] = [0.0, 0.0, angle + np.pi]

        return start_pos, start_orn

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset the environment with randomized spawn positions."""
        if options is None:
            options = {}

        # Seed RNG
        if seed is not None:
            self._np_random = np.random.default_rng(seed)

        # Generate spawn positions
        self.start_pos, self.start_orn = self._get_start_pos_orn(seed)

        # Reset aviary
        super().begin_reset(seed, options)
        super().end_reset(seed, options)

        # Visual rendering: team colors + dome wireframe
        if self.render_mode:
            self._color_teams()
            self._draw_dome()

        # Reset game state
        self.captured = np.zeros(self.num_possible_agents, dtype=bool)
        self.capture_happened = False
        self.prev_min_dist_to_evader = np.full(self.num_pursuers, np.inf)

        # Compute initial pairwise distances
        self._update_pairwise_distances()

        # Build initial observations
        observations = {
            ag: self.compute_observation_by_id(self.agent_name_mapping[ag])
            for ag in self.agents
        }
        infos = {ag: dict() for ag in self.agents}
        return observations, infos

    def _color_teams(self):
        """Color pursuers blue and evaders red in the PyBullet renderer."""
        pursuer_color = [0.2, 0.4, 1.0, 1.0]  # blue
        evader_color = [1.0, 0.2, 0.2, 1.0]  # red
        p = self.aviary

        for agent_id in range(self.num_possible_agents):
            drone_id = self.aviary.drones[agent_id].Id
            color = evader_color if self.is_evader[agent_id] else pursuer_color

            # Color the base link (-1)
            p.changeVisualShape(drone_id, -1, rgbaColor=color)
            # Color all child links
            for link_id in range(p.getNumJoints(drone_id)):
                p.changeVisualShape(drone_id, link_id, rgbaColor=color)

    def _draw_dome(self):
        """Draw a wireframe sphere representing the arena boundary."""
        self._dome_line_ids = draw_dome(
            self.aviary, self.flight_dome_size, self._dome_line_ids
        )

    def _update_pairwise_distances(self):
        """Compute NxN pairwise distance matrix from current positions."""
        positions = np.array(
            [self.aviary.state(i)[-1] for i in range(self.num_possible_agents)]
        )
        diff = positions[:, None, :] - positions[None, :, :]
        self.pairwise_distances = np.linalg.norm(diff, axis=-1)

    def update_states(self):
        """Called once per physics substep. Precompute shared state."""
        self._update_pairwise_distances()
        self._check_captures()

        # Cache shared info once per substep (avoid rebuilding per agent)
        self._cached_info = {
            "agent_positions": {
                i: self.aviary.state(i)[-1].copy() for i in range(self.num_possible_agents)
            },
            "agent_velocities": {
                i: self.aviary.state(i)[2].copy() for i in range(self.num_possible_agents)
            },
            "pairwise_distances": self.pairwise_distances.copy(),
            "is_evader": self.is_evader,
            "captured": self.captured.copy(),
            "capture_event": self.capture_happened,
        }

    def _check_captures(self):
        """Check if any pursuer has captured any evader."""
        self.capture_happened = False
        for e_id in self.evader_ids:
            if self.captured[e_id]:
                continue
            for p_id in self.pursuer_ids:
                if self.pairwise_distances[p_id, e_id] < self.capture_distance:
                    self.captured[e_id] = True
                    self.capture_happened = True
                    break

    def compute_observation_by_id(self, agent_id: int) -> np.ndarray:
        """Build flat observation for agent_id.

        Layout: [self_state(20) | teammate(6) | opponent_0(7) | opponent_1(7) | boundary(1)]
        """
        ang_vel, ang_pos, lin_vel, lin_pos, quaternion = self.compute_attitude_by_id(
            agent_id
        )
        aux_state = self.aviary.aux_state(agent_id)

        # Self-state (20 for euler, 21 for quaternion)
        if self.angle_representation == 0:  # euler
            self_obs = np.concatenate(
                [ang_vel, ang_pos, lin_vel, lin_pos, aux_state, self.past_actions[agent_id]]
            )
        else:  # quaternion
            self_obs = np.concatenate(
                [ang_vel, quaternion, lin_vel, lin_pos, aux_state, self.past_actions[agent_id]]
            )

        agent_is_evader = self.is_evader[agent_id]
        teammate_ids = self._teammate_ids[agent_id]
        opponent_ids = self._opponent_ids[agent_id]
        vision_dist = (
            self.vision_distance_evader if agent_is_evader
            else self.vision_distance_pursuer
        )

        # Teammate relative state (6 per teammate)
        teammate_obs_parts = []
        for t_id in teammate_ids:
            t_state = self.aviary.state(t_id)
            t_vel = t_state[2]
            t_pos = t_state[3]
            rel_pos = t_pos - lin_pos
            rel_vel = t_vel - lin_vel
            teammate_obs_parts.append(np.concatenate([rel_pos, rel_vel]))

        if len(teammate_obs_parts) > 0:
            teammate_obs = np.concatenate(teammate_obs_parts)
        else:
            teammate_obs = np.array([], dtype=np.float64)

        # Opponent relative state (7 per opponent: vision_flag + rel_pos + rel_vel)
        opponent_obs_parts = []
        for o_id in opponent_ids:
            dist = self.pairwise_distances[agent_id, o_id]
            in_vision = dist <= vision_dist and not self.captured[o_id]

            if in_vision:
                o_state = self.aviary.state(o_id)
                o_vel = o_state[2]
                o_pos = o_state[3]
                rel_pos = o_pos - lin_pos
                rel_vel = o_vel - lin_vel
                opponent_obs_parts.append(
                    np.concatenate([[1.0], rel_pos, rel_vel])
                )
            else:
                opponent_obs_parts.append(np.zeros(7))

        if len(opponent_obs_parts) > 0:
            opponent_obs = np.concatenate(opponent_obs_parts)
        else:
            opponent_obs = np.array([], dtype=np.float64)

        # Boundary info (1 dim: normalized distance from origin)
        dist_from_origin = np.linalg.norm(lin_pos)
        boundary_obs = np.array([dist_from_origin / self.flight_dome_size])

        return np.concatenate([self_obs, teammate_obs, opponent_obs, boundary_obs])

    def compute_term_trunc_reward_info_by_id(
        self, agent_id: int
    ) -> tuple[bool, bool, float, dict[str, Any]]:
        """Compute termination, truncation, reward, and info for one agent."""
        reward = 0.0
        term = False
        trunc = self.step_count > self.max_steps
        info: dict[str, Any] = {}

        lin_pos = self.aviary.state(agent_id)[-1]
        dist_from_origin = np.linalg.norm(lin_pos)
        agent_is_evader = self.is_evader[agent_id]

        rwd: dict[str, float] = {}

        # --- Collision check ---
        if np.any(self.aviary.contact_array[self.aviary.drones[agent_id].Id]):
            rwd["collision"] = -self.collision_penalty
            info["collision"] = True
            term = True

        # --- Boundary enforcement (3 zones) ---
        grey_zone_start = self.flight_dome_size * self.boundary_penalty_fraction
        if dist_from_origin > self.flight_dome_size:
            # Red zone: hard termination
            rwd["oob"] = -self.collision_penalty
            info["out_of_bounds"] = True
            term = True
        elif dist_from_origin > grey_zone_start:
            # Grey zone: linearly increasing penalty
            grey_zone_width = self.flight_dome_size - grey_zone_start
            penetration = (dist_from_origin - grey_zone_start) / grey_zone_width
            rwd["boundary"] = -self.boundary_penalty_scale * penetration

        # --- Capture mechanics ---
        if self.capture_happened and self.terminate_on_first_capture:
            if agent_is_evader:
                if self.captured[agent_id]:
                    rwd["captured"] = -self.captured_penalty
                    info["captured"] = True
                else:
                    rwd["teammate_captured"] = -self.teammate_captured_penalty
                    info["teammate_captured"] = True
            else:
                # Check if this pursuer was the one who captured
                for e_id in self.evader_ids:
                    if (
                        self.captured[e_id]
                        and self.pairwise_distances[agent_id, e_id]
                        < self.capture_distance
                    ):
                        rwd["capture"] = self.capture_reward
                        info["made_capture"] = True
                        break
                else:
                    rwd["team_capture"] = self.team_capture_reward
                    info["team_capture"] = True

                if np.all(self.captured[self.evader_ids]):
                    rwd["all_captured"] = self.all_captured_reward
                    info["all_captured"] = True

            term = True

        # --- Dense rewards (only if not terminated) ---
        if not term and not trunc:
            if agent_is_evader:
                rwd["survival"] = self.survival_reward_per_step

                dists_to_pursuers = self.pairwise_distances[
                    agent_id, self.pursuer_ids
                ]
                min_dist = np.min(dists_to_pursuers)
                rwd["evader_distance"] = self.evader_distance_reward_scale * (
                    min_dist / self.flight_dome_size
                )

            else:
                rwd["time_penalty"] = -self.time_penalty

                active_evader_ids = self.evader_ids[~self.captured[self.evader_ids]]
                if len(active_evader_ids) > 0:
                    dists_to_evaders = self.pairwise_distances[
                        agent_id, active_evader_ids
                    ]
                    min_dist = np.min(dists_to_evaders)

                    p_idx = self._pursuer_id_to_idx[agent_id]
                    prev_dist = self.prev_min_dist_to_evader[p_idx]

                    if prev_dist < np.inf:
                        rwd["distance_progress"] = (
                            self.distance_reward_scale * (prev_dist - min_dist)
                        )

                    self.prev_min_dist_to_evader[p_idx] = min_dist

        # --- Truncation rewards ---
        if trunc and not term:
            if agent_is_evader and not self.captured[agent_id]:
                rwd["all_survived"] = self.all_survived_reward
                info["survived"] = True

        reward = sum(rwd.values())
        info["reward_components"] = rwd

        # Attach cached shared info (built once per substep in update_states)
        info.update(self._cached_info)

        return term, trunc, reward, info

    def capture_frame(self) -> np.ndarray:
        """Return an RGB frame from a top-down debug visualizer camera."""
        return _capture_frame(
            self.aviary,
            self.flight_dome_size,
            width=self._frame_width,
            height=self._frame_height,
        )

    def interpret_outcome(self, infos: dict[str, dict[str, Any]]) -> str:
        """Interpret episode outcome from the final step's info dicts.

        Returns:
            "pursuers_win" — at least one evader was captured.
            "evaders_win"  — time ran out with no captures.
            "draw"         — both sides terminated for other reasons (OOB, collision).
        """
        any_captured = np.any(self.captured[self.evader_ids])
        any_survived = any(
            infos.get(ag, {}).get("survived", False) for ag in infos
        )

        if any_captured:
            return "pursuers_win"
        if any_survived:
            return "evaders_win"
        return "draw"

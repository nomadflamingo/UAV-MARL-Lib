"""QuadX Dome Waypoints Environment.

Single-agent environment where a quadrotor collects waypoints one at a time
inside a spherical dome arena. A new waypoint spawns at a random location
after each collection. Episode terminates on dome exit or floor collision.
"""

from __future__ import annotations

import os
from typing import Any, Literal

import numpy as np
from gymnasium import spaces

from PyFlyt.gym_envs.quadx_envs.quadx_base_env import QuadXBaseEnv
from PyFlyt.gym_envs.utils.dome_renderer import capture_frame as _capture_frame
from PyFlyt.gym_envs.utils.dome_renderer import draw_dome


class QuadXDomeWaypointsEnv(QuadXBaseEnv):
    """QuadX Dome Waypoints Environment.

    Actions are vp, vq, vr, T, ie: angular rates and thrust.
    The agent collects waypoints one at a time inside a spherical dome.

    Args:
        sparse_reward (bool): whether to use sparse rewards or not.
        waypoint_reach_distance (float): distance threshold for collecting a waypoint.
        waypoint_reward (float): reward for collecting a waypoint.
        flight_dome_size (float): radius of the spherical arena.
        time_penalty (float): penalty per step (encourages fast collection).
        distance_reward_scale (float): scale for dense approach reward (0 when sparse).
        spawn_min_height (float): minimum height for spawned waypoints.
        spawn_max_radius_fraction (float): waypoints spawn within this fraction of dome radius.
        flight_mode (int): the flight mode of the UAV.
        max_duration_seconds (float): maximum simulation time of the environment.
        angle_representation (Literal["euler", "quaternion"]): angle representation.
        agent_hz (int): looprate of the agent to environment interaction.
        render_mode (None | Literal["human", "rgb_array"]): render_mode.
        render_resolution (tuple[int, int]): render_resolution.
    """

    def __init__(
        self,
        sparse_reward: bool = False,
        waypoint_reach_distance: float = 0.5,
        waypoint_reward: float = 100.0,
        flight_dome_size: float = 8.0,
        time_penalty: float = 0.1,
        distance_reward_scale: float = 1.0,
        spawn_min_height: float = 0.5,
        spawn_max_radius_fraction: float = 0.9,
        flight_mode: int = 0,
        max_duration_seconds: float = 30.0,
        angle_representation: Literal["euler", "quaternion"] = "euler",
        agent_hz: int = 30,
        render_mode: None | Literal["human", "rgb_array"] = None,
        render_resolution: tuple[int, int] = (480, 480),
    ):
        super().__init__(
            start_pos=np.array([[0.0, 0.0, 1.0]]),
            flight_mode=flight_mode,
            flight_dome_size=flight_dome_size,
            max_duration_seconds=max_duration_seconds,
            angle_representation=angle_representation,
            agent_hz=agent_hz,
            render_mode=render_mode,
            render_resolution=render_resolution,
        )

        self.sparse_reward = sparse_reward
        self.waypoint_reach_distance = waypoint_reach_distance
        self.waypoint_reward = waypoint_reward
        self.time_penalty = time_penalty
        self.distance_reward_scale = distance_reward_scale
        self.spawn_min_height = spawn_min_height
        self.spawn_max_radius = flight_dome_size * spawn_max_radius_fraction

        # Observation: attitude + waypoint_delta(3) + boundary_dist(1)
        obs_size = self.combined_space.shape[0] + 3 + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        # Waypoint visual URDF path
        file_dir = os.path.dirname(os.path.realpath(__file__))
        self._targ_obj_dir = os.path.join(file_dir, "../../models/target.urdf")

        # Runtime state (initialized properly in reset)
        self.waypoint_pos = np.zeros(3)
        self.num_waypoints_collected = 0
        self._prev_distance = np.inf
        self._waypoint_collected_this_step = False
        self._dome_line_ids: list[int] = []
        self._waypoint_visual_id: int | None = None

    def _spawn_waypoint(self) -> np.ndarray:
        """Sample a random waypoint position inside the dome safe zone."""
        lin_pos = self.env.state(0)[-1] if hasattr(self, "env") else None

        for _ in range(50):
            # Uniform sampling in spherical volume
            r = self.spawn_max_radius * (self.np_random.uniform() ** (1.0 / 3.0))
            theta = self.np_random.uniform(0.0, 2.0 * np.pi)
            phi = np.arccos(1.0 - 2.0 * self.np_random.uniform())
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = abs(r * np.cos(phi))
            z = max(z, self.spawn_min_height)
            pos = np.array([x, y, z])

            # Reject if too close to the drone
            if lin_pos is not None:
                if np.linalg.norm(pos - lin_pos) < self.waypoint_reach_distance * 2.0:
                    continue
            return pos

        # Fallback (should rarely happen)
        return pos

    def _draw_waypoint(self):
        """Load waypoint visual at current waypoint position."""
        self._waypoint_visual_id = self.env.loadURDF(
            self._targ_obj_dir,
            basePosition=self.waypoint_pos.tolist(),
            useFixedBase=True,
            globalScaling=self.waypoint_reach_distance / 2.0,
        )
        self.env.changeVisualShape(
            self._waypoint_visual_id,
            linkIndex=-1,
            rgbaColor=[0.0, 1.0, 0.2, 0.8],
        )

    def _remove_waypoint_visual(self):
        """Remove previous waypoint visual."""
        if self._waypoint_visual_id is not None:
            self.env.removeBody(self._waypoint_visual_id)
            self._waypoint_visual_id = None

    def reset(
        self, *, seed: None | int = None, options: None | dict[str, Any] = dict()
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment."""
        # Randomise start position using a seed-derived RNG so it's reproducible.
        # We use a separate RNG so self.np_random (set inside begin_reset) is
        # unaffected, keeping waypoint spawning deterministic with the same seed.
        pos_rng = np.random.default_rng(seed)
        r = self.spawn_max_radius * (pos_rng.uniform() ** (1.0 / 3.0))
        theta = pos_rng.uniform(0.0, 2.0 * np.pi)
        start_z = pos_rng.uniform(self.spawn_min_height + 0.5, self.spawn_max_radius * 0.4)
        self.start_pos = np.array([[r * np.cos(theta), r * np.sin(theta), start_z]])
        self.start_orn = np.array([[0.0, 0.0, theta + np.pi]])

        super().begin_reset(seed, options)

        # Spawn first waypoint
        self.waypoint_pos = self._spawn_waypoint()
        self.num_waypoints_collected = 0
        self._prev_distance = np.inf
        self._waypoint_collected_this_step = False
        self._dome_line_ids = []
        self._waypoint_visual_id = None

        super().end_reset(seed, options)

        # Visuals work headlessly via getCameraImage — always draw
        self._dome_line_ids = draw_dome(self.env, self.flight_dome_size)
        self._draw_waypoint()

        self.info["num_waypoints_collected"] = 0
        return self.state, self.info

    def compute_state(self) -> None:
        """Compute the observation vector."""
        ang_vel, ang_pos, lin_vel, lin_pos, quaternion = super().compute_attitude()
        aux_state = super().compute_auxiliary()

        # Cache for reuse in compute_term_trunc_reward (avoids redundant PyBullet query)
        self._lin_pos = lin_pos

        # Attitude block
        if self.angle_representation == 0:
            attitude = np.concatenate(
                [ang_vel, ang_pos, lin_vel, lin_pos, self.action, aux_state]
            )
        else:
            attitude = np.concatenate(
                [ang_vel, quaternion, lin_vel, lin_pos, self.action, aux_state]
            )

        # Waypoint delta (world frame)
        waypoint_delta = self.waypoint_pos - lin_pos

        # Boundary distance (0 at center, 1 at dome edge)
        dist_from_origin = np.linalg.norm(lin_pos)
        boundary_dist = np.array([dist_from_origin / self.flight_dome_size])

        self.state = np.concatenate([attitude, waypoint_delta, boundary_dist])

    def compute_term_trunc_reward(self) -> None:
        """Compute termination, truncation, and reward."""
        # Base handles: floor collision, OOB at flight_dome_size, step limit
        super().compute_base_term_trunc_reward()

        if self.termination:
            return

        # Waypoint collection check (lin_pos cached by compute_state)
        lin_pos = self._lin_pos
        waypoint_distance = np.linalg.norm(lin_pos - self.waypoint_pos)

        if waypoint_distance < self.waypoint_reach_distance:
            self._waypoint_collected_this_step = True
            self.num_waypoints_collected += 1
            self.info["num_waypoints_collected"] = self.num_waypoints_collected

            self._remove_waypoint_visual()
            self.waypoint_pos = self._spawn_waypoint()
            self._draw_waypoint()
            self._prev_distance = np.inf  # reset; updated below to distance to NEW waypoint

        # Dense distance reward — always measured against the CURRENT waypoint
        # (which may have just been updated). This ensures _prev_distance and the
        # current distance are both relative to the same target.
        if not self.sparse_reward and self.distance_reward_scale > 0.0:
            current_distance = np.linalg.norm(lin_pos - self.waypoint_pos)
            if not np.isinf(self._prev_distance):
                progress = self._prev_distance - current_distance
                self._distance_progress += self.distance_reward_scale * progress
                self.reward += self.distance_reward_scale * progress
            self._prev_distance = current_distance

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Step the environment."""
        self.action = action.copy()
        self.reward = -self.time_penalty
        self._waypoint_collected_this_step = False
        self._distance_progress = 0.0
        self.env.set_setpoint(0, action)

        for _ in range(self.env_step_ratio):
            if self.termination or self.truncation:
                break
            self.env.step()
            self.compute_state()
            self.compute_term_trunc_reward()

        # Apply waypoint bonus after the loop so sub-steps don't overwrite it
        if self._waypoint_collected_this_step:
            self.reward += self.waypoint_reward

        # Build reward_components for WandB logging
        rwd: dict[str, float] = {"time_penalty": -self.time_penalty}
        if self._waypoint_collected_this_step:
            rwd["waypoint"] = self.waypoint_reward
        if self._distance_progress != 0.0:
            rwd["distance_progress"] = self._distance_progress
        self.info["reward_components"] = rwd
        self.info["num_waypoints_collected"] = self.num_waypoints_collected

        self.step_count += 1
        return self.state, self.reward, self.termination, self.truncation, self.info

    def capture_frame(self) -> np.ndarray:
        """Return an RGB frame from a fixed top-down camera (works headlessly)."""
        return _capture_frame(self.env, self.flight_dome_size)

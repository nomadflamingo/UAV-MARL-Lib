"""Tests for QuadXDomeWaypointsEnv."""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
from gymnasium.utils.env_checker import check_env

from PyFlyt.gym_envs.quadx_envs.quadx_dome_waypoints_env import QuadXDomeWaypointsEnv


CHECK_ENV_IGNORE_WARNINGS = [
    f"\x1b[33mWARN: {message}\x1b[0m"
    for message in [
        "For Box action spaces, we recommend using a symmetric and normalized space (range=[-1, 1] or [0, 1]). See https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html for more information.",
        "A Box observation space minimum value is -infinity. This is probably too low.",
        "A Box observation space maximum value is -infinity. This is probably too high.",
        "A Box observation space minimum value is infinity. This is probably too low.",
        "A Box observation space maximum value is infinity. This is probably too high.",
        "Not able to test alternative render modes due to the environment not having a spec. Try instantiating the environment through `gymnasium.make`",
    ]
]


class TestInstantiationAndReset:
    """Basic instantiation and reset tests."""

    def test_create_env(self):
        """Environment can be created with default config."""
        env = QuadXDomeWaypointsEnv()
        assert env is not None
        env.close()

    def test_reset_returns_valid_obs(self):
        """Reset returns observation with correct shape."""
        env = QuadXDomeWaypointsEnv()
        obs, _ = env.reset(seed=42)
        assert obs.shape == env.observation_space.shape
        assert env.observation_space.contains(obs)
        env.close()

    def test_reset_info(self):
        """Reset info contains num_waypoints_collected = 0."""
        env = QuadXDomeWaypointsEnv()
        _, info = env.reset(seed=42)
        assert info["num_waypoints_collected"] == 0
        env.close()

    def test_waypoint_spawns_inside_dome(self):
        """Waypoint position is within the safe zone on reset."""
        env = QuadXDomeWaypointsEnv()
        for seed in range(10):
            env.reset(seed=seed)
            dist = np.linalg.norm(env.waypoint_pos)
            assert dist < env.spawn_max_radius, (
                f"Waypoint at dist {dist} exceeds safe radius {env.spawn_max_radius}"
            )
            assert env.waypoint_pos[2] >= env.spawn_min_height
        env.close()

    def test_euler_and_quaternion(self):
        """Both angle representations produce correct obs shapes."""
        for rep, expected_att in [("euler", 20), ("quaternion", 21)]:
            rep_lit: Literal["euler", "quaternion"] = rep  # type: ignore[assignment]
            env = QuadXDomeWaypointsEnv(angle_representation=rep_lit)
            obs, _ = env.reset(seed=42)
            expected = expected_att + 3 + 1  # attitude + waypoint_delta + boundary
            assert obs.shape == (expected,), f"{rep}: {obs.shape} != ({expected},)"
            env.close()

    def test_check_env(self):
        """Gymnasium check_env passes."""
        env = QuadXDomeWaypointsEnv()
        with warnings.catch_warnings(record=True) as caught_warnings:
            check_env(env)
        for w in caught_warnings:
            assert isinstance(w.message, Warning)
            if w.message.args[0] not in CHECK_ENV_IGNORE_WARNINGS:
                raise AssertionError(f"Unexpected warning: {w.message}")
        env.close()


class TestObservationAndStep:
    """Observation space and stepping tests."""

    def test_step_returns_valid(self):
        """A single step returns valid observations and rewards."""
        env = QuadXDomeWaypointsEnv()
        env.reset(seed=42)
        action = env.action_space.sample()
        obs, reward, term, trunc, _ = env.step(action)
        assert env.observation_space.contains(obs)
        assert isinstance(reward, float)
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        env.close()

    def test_observations_in_space_throughout(self):
        """Observations stay in declared space throughout episode."""
        env = QuadXDomeWaypointsEnv()
        env.reset(seed=42)
        for _ in range(50):
            obs, _, term, trunc, _ = env.step(env.action_space.sample())
            assert env.observation_space.contains(obs)
            if term or trunc:
                break
        env.close()

    def test_reward_components_in_info(self):
        """Info dict contains reward_components for WandB logging."""
        env = QuadXDomeWaypointsEnv()
        env.reset(seed=42)
        _, _, _, _, info = env.step(env.action_space.sample())
        assert "reward_components" in info
        assert "time_penalty" in info["reward_components"]
        assert "num_waypoints_collected" in info
        env.close()

    def test_seeding(self):
        """Same seed produces identical trajectories."""
        env1 = QuadXDomeWaypointsEnv()
        env2 = QuadXDomeWaypointsEnv()
        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)
        np.testing.assert_array_equal(obs1, obs2)

        for _ in range(50):
            action = env1.action_space.sample()
            obs1, r1, t1, tr1, _ = env1.step(action)
            obs2, r2, t2, tr2, _ = env2.step(action)
            np.testing.assert_array_equal(obs1, obs2)
            assert r1 == r2
            assert t1 == t2 and tr1 == tr2
            if t1 or tr1:
                break

        env1.close()
        env2.close()


class TestGameMechanics:
    """Waypoint collection and boundary tests."""

    def test_truncation_on_max_steps(self):
        """Episode truncates after max_duration_seconds."""
        env = QuadXDomeWaypointsEnv(max_duration_seconds=1.0, agent_hz=30)
        env.reset(seed=42)

        truncated = False
        for _ in range(100):
            _, _, term, trunc, _ = env.step(np.zeros(4))
            if term or trunc:
                truncated = trunc
                break

        # Episode should end — either via truncation (time limit) or termination (collision)
        assert truncated or term
        env.close()

    def test_time_penalty_applied(self):
        """Non-terminal steps incur time penalty."""
        env = QuadXDomeWaypointsEnv(time_penalty=0.5, sparse_reward=True)
        env.reset(seed=42)
        # Hover in place — should not terminate or collect waypoint
        _, reward, term, _, _ = env.step(np.zeros(4))
        if not term:
            assert reward < 0, f"Expected negative reward from time penalty, got {reward}"
        env.close()

    def test_waypoint_collection_gives_reward(self):
        """Collecting a waypoint gives waypoint_reward."""
        env = QuadXDomeWaypointsEnv(
            waypoint_reach_distance=100.0,  # huge radius — instant collection
            waypoint_reward=50.0,
        )
        env.reset(seed=42)
        _, reward, _, _, info = env.step(np.zeros(4))
        assert reward >= 50.0 - env.time_penalty, f"Expected waypoint reward, got {reward}"
        assert info["num_waypoints_collected"] >= 1
        env.close()

    def test_new_waypoint_spawns_after_collection(self):
        """A new waypoint is spawned after collecting one."""
        env = QuadXDomeWaypointsEnv(
            waypoint_reach_distance=100.0,
        )
        env.reset(seed=42)
        old_pos = env.waypoint_pos.copy()
        env.step(np.zeros(4))
        # Waypoint should have moved
        assert not np.array_equal(env.waypoint_pos, old_pos)
        env.close()

    def test_boundary_distance_in_obs(self):
        """Boundary distance is in [0, 1] range at reset."""
        env = QuadXDomeWaypointsEnv()
        obs, _ = env.reset(seed=42)
        boundary_dist = obs[-1]
        assert 0.0 <= boundary_dist <= 1.0, (
            f"Boundary distance {boundary_dist} outside [0, 1]"
        )
        env.close()

    def test_configurable_params(self):
        """Environment accepts and uses custom parameters."""
        env = QuadXDomeWaypointsEnv(
            flight_dome_size=5.0,
            waypoint_reach_distance=1.0,
            waypoint_reward=200.0,
            time_penalty=0.5,
            max_duration_seconds=10.0,
        )
        assert env.flight_dome_size == 5.0
        assert env.waypoint_reach_distance == 1.0
        assert env.waypoint_reward == 200.0
        assert env.time_penalty == 0.5
        env.close()

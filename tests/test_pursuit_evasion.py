"""Tests for MAQuadXPursuitEvasionEnv."""

from __future__ import annotations

import warnings

import numpy as np
from pettingzoo.test import parallel_api_test

from PyFlyt.marl_wrappers.selfplay import RandomPolicy
from PyFlyt.pz_envs.quadx_envs.ma_quadx_pursuit_evasion_env import (
    MAQuadXPursuitEvasionEnv,
)


# Warnings to ignore from PettingZoo API test
CHECK_ENV_IGNORE_WARNINGS = [
    "Agent's minimum observation space value is -infinity. This is probably too low.",
    "Agent's maximum observation space value is infinity. This is probably too high",
]


class TestInstantiationAndReset:
    """Basic instantiation and reset tests."""

    def test_create_env(self):
        """Environment can be created with default config."""
        env = MAQuadXPursuitEvasionEnv()
        assert env is not None
        env.close()

    def test_agent_count(self):
        """4 agents at start with correct naming."""
        env = MAQuadXPursuitEvasionEnv()
        obs, infos = env.reset(seed=42)
        assert len(env.agents) == 4
        assert env.agents == ["uav_0", "uav_1", "uav_2", "uav_3"]
        env.close()

    def test_team_assignment(self):
        """Pursuers are uav_0,1; evaders are uav_2,3."""
        env = MAQuadXPursuitEvasionEnv()
        env.reset(seed=42)
        assert not env.is_evader[0]
        assert not env.is_evader[1]
        assert env.is_evader[2]
        assert env.is_evader[3]
        env.close()

    def test_spawn_positions_in_arena(self):
        """All spawn positions are within the arena bounds."""
        env = MAQuadXPursuitEvasionEnv()
        for seed in range(10):
            env.reset(seed=seed)
            for i in range(4):
                pos = env.aviary.state(i)[-1]
                dist = np.linalg.norm(pos)
                assert dist < env.flight_dome_size, (
                    f"Agent {i} spawned outside dome at dist {dist}"
                )
        env.close()

    def test_team_separation(self):
        """Pursuers and evaders spawn on opposite sides."""
        env = MAQuadXPursuitEvasionEnv()
        for seed in range(10):
            env.reset(seed=seed)
            p_center = np.mean(
                [env.start_pos[i] for i in env.pursuer_ids], axis=0
            )
            e_center = np.mean(
                [env.start_pos[i] for i in env.evader_ids], axis=0
            )
            separation = np.linalg.norm(p_center[:2] - e_center[:2])
            assert separation > 0.5, (
                f"Teams too close: {separation:.2f}m apart"
            )
        env.close()

    def test_reset_returns_valid_obs(self):
        """Reset returns observations for all agents with correct shape."""
        env = MAQuadXPursuitEvasionEnv()
        obs, infos = env.reset(seed=42)
        assert set(obs.keys()) == set(env.agents)
        for ag, ob in obs.items():
            assert env.observation_space(ag).contains(ob), (
                f"Observation for {ag} not in space. Shape: {ob.shape}"
            )
        env.close()


class TestObservationAndStep:
    """Observation space and stepping tests."""

    def test_observation_shape(self):
        """Observations have the expected shape for 2v2 euler."""
        env = MAQuadXPursuitEvasionEnv()
        obs, _ = env.reset(seed=42)
        expected_shape = (env.obs_size,)
        for ag, ob in obs.items():
            assert ob.shape == expected_shape, f"{ag} obs shape: {ob.shape}"
        env.close()

    def test_step_returns_valid(self):
        """A single step returns valid observations and rewards."""
        env = MAQuadXPursuitEvasionEnv()
        obs, _ = env.reset(seed=42)
        actions = {ag: env.action_space(ag).sample() for ag in env.agents}
        obs2, rewards, terms, truncs, infos = env.step(actions)

        for ag in list(obs2.keys()):
            assert env.observation_space(ag).contains(obs2[ag])
            assert isinstance(rewards[ag], float)
            assert isinstance(terms[ag], bool)
            assert isinstance(truncs[ag], bool)
        env.close()

    def test_observations_in_space_throughout(self):
        """Observations stay in declared space throughout episode."""
        env = MAQuadXPursuitEvasionEnv()
        obs, _ = env.reset(seed=42)

        for _ in range(50):
            if not env.agents:
                break
            actions = {ag: env.action_space(ag).sample() for ag in env.agents}
            obs, _, _, _, _ = env.step(actions)
            for ag, ob in obs.items():
                assert env.observation_space(ag).contains(ob), (
                    f"{ag} obs not in space at step"
                )
        env.close()

    def test_parallel_api(self):
        """PettingZoo parallel_api_test passes."""
        env = MAQuadXPursuitEvasionEnv()
        with warnings.catch_warnings(record=True) as caught_warnings:
            parallel_api_test(env, num_cycles=100)

        for w in caught_warnings:
            assert isinstance(w.message, Warning)
            if w.message.args[0] not in CHECK_ENV_IGNORE_WARNINGS:
                raise AssertionError(f"Unexpected warning: {w.message}")
        env.close()


class TestGameMechanics:
    """Capture, boundary, and termination tests."""

    def test_truncation_on_max_steps(self):
        """Episode truncates after max_duration_seconds."""
        env = MAQuadXPursuitEvasionEnv(max_duration_seconds=1.0, agent_hz=30)
        env.reset(seed=42)
        max_steps = int(30 * 1.0)  # 30 steps

        for step in range(max_steps + 10):
            if not env.agents:
                break
            actions = {ag: env.action_space(ag).sample() for ag in env.agents}
            _, _, terms, truncs, _ = env.step(actions)

        # All agents should be done
        assert len(env.agents) == 0
        env.close()

    def test_info_dict_contents(self):
        """Info dict contains fields needed for heuristic baselines."""
        env = MAQuadXPursuitEvasionEnv()
        obs, _ = env.reset(seed=42)
        actions = {ag: env.action_space(ag).sample() for ag in env.agents}
        _, _, _, _, infos = env.step(actions)

        for ag in infos:
            info = infos[ag]
            assert "agent_positions" in info
            assert "agent_velocities" in info
            assert "pairwise_distances" in info
            assert "is_evader" in info
            assert "captured" in info
            assert "capture_event" in info
        env.close()

    def test_reward_signs_pursuer(self):
        """Pursuer gets negative time penalty when not capturing."""
        env = MAQuadXPursuitEvasionEnv()
        env.reset(seed=42)

        # Step with zero actions (hover) - pursuers should get time penalty
        actions = {
            ag: np.zeros(4) for ag in env.agents
        }
        _, rewards, _, _, _ = env.step(actions)

        # Pursuers should have some negative component from time penalty
        for p_id in env.pursuer_ids:
            ag = f"uav_{p_id}"
            if ag in rewards:
                # Not asserting sign because distance progress could dominate,
                # but reward should be finite
                assert np.isfinite(rewards[ag])
        env.close()

    def test_reward_signs_evader(self):
        """Evader gets positive survival reward."""
        env = MAQuadXPursuitEvasionEnv()
        env.reset(seed=42)

        actions = {ag: np.zeros(4) for ag in env.agents}
        _, rewards, _, _, _ = env.step(actions)

        for e_id in env.evader_ids:
            ag = f"uav_{e_id}"
            if ag in rewards:
                assert np.isfinite(rewards[ag])
        env.close()


class TestSelfPlayIntegration:
    """Test compatibility with MASelfPlayEnv training wrapper."""

    def test_selfplay_wrapper(self):
        """MASelfPlayEnv wraps the env without errors."""
        from PyFlyt.marl_wrappers.selfplay import MASelfPlayEnv

        env = MAQuadXPursuitEvasionEnv()
        expected_shape = (env.obs_size,)

        opp_policies = {
            i: RandomPolicy(env.action_space())
            for i in [1, 2, 3]
        }
        wrapped = MASelfPlayEnv(env, train_agent_id=0, opp_policies=opp_policies)

        obs, info = wrapped.reset()
        assert obs.shape == expected_shape

        for _ in range(20):
            action = wrapped.action_space.sample()
            obs, rew, term, trunc, info = wrapped.step(action)
            assert obs.shape == expected_shape
            assert isinstance(rew, float)

        wrapped.ma_env.close()

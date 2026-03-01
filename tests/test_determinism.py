"""
Determinism tests — verify that identical seeds produce identical behaviour.

These tests are not about perfect bitwise reproducibility across platforms;
they verify that within a single machine:
  - The same seed → same initial observations.
  - The same seed → same policy outputs (given same weights).
  - GAE is deterministic (pure NumPy, no randomness).
  - A short rollout with a fixed seed produces a consistent trajectory.

Runtime target: < 30 seconds on CPU.
"""

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Environment seeding
# ---------------------------------------------------------------------------

class TestEnvSeeding:
    """Same seed produces same initial observation across separate env instances."""

    def test_state_obs_reproducible(self):
        from visual_rl_locomotion.envs.make_env import make_env
        e1 = make_env("Hopper-v4", seed=42, obs_mode="state")
        e2 = make_env("Hopper-v4", seed=42, obs_mode="state")
        o1, _ = e1.reset(seed=42)
        o2, _ = e2.reset(seed=42)
        e1.close()
        e2.close()
        np.testing.assert_array_equal(o1, o2,
            err_msg="State observations differ for identical seeds.")

    def test_pixel_obs_reproducible(self):
        from visual_rl_locomotion.envs.make_env import make_env
        e1 = make_env("Hopper-v4", seed=7, obs_mode="pixels", img_size=64)
        e2 = make_env("Hopper-v4", seed=7, obs_mode="pixels", img_size=64)
        o1, _ = e1.reset(seed=7)
        o2, _ = e2.reset(seed=7)
        e1.close()
        e2.close()
        np.testing.assert_array_equal(o1, o2,
            err_msg="Pixel observations differ for identical seeds.")

    def test_different_seeds_differ(self):
        from visual_rl_locomotion.envs.make_env import make_env
        e1 = make_env("Hopper-v4", seed=0, obs_mode="state")
        e2 = make_env("Hopper-v4", seed=99, obs_mode="state")
        o1, _ = e1.reset(seed=0)
        o2, _ = e2.reset(seed=99)
        e1.close()
        e2.close()
        # It would be extremely unlikely for seed-0 and seed-99 to give the
        # same initial state; this verifies seeding actually varies output.
        assert not np.array_equal(o1, o2), \
            "Different seeds produced identical initial observations."


# ---------------------------------------------------------------------------
# Policy determinism
# ---------------------------------------------------------------------------

class TestPolicyDeterminism:
    """Deterministic mode yields identical actions for identical inputs."""

    def test_mlp_policy_deterministic_action(self):
        from visual_rl_locomotion.models.mlp_policy import MLPPolicy
        from visual_rl_locomotion.utils.seed import set_seed

        set_seed(0)
        policy = MLPPolicy(obs_dim=11, action_dim=3)
        obs = torch.zeros(1, 11)

        a1, _, _ = policy.get_action_and_logp(obs, deterministic=True)
        a2, _, _ = policy.get_action_and_logp(obs, deterministic=True)
        torch.testing.assert_close(a1, a2,
            msg="Deterministic MLP policy should return identical actions.")

    def test_vision_policy_deterministic_action(self):
        from visual_rl_locomotion.models.vision_policy import VisionPolicy
        from visual_rl_locomotion.utils.seed import set_seed

        set_seed(0)
        policy = VisionPolicy(obs_shape=(3, 64, 64), action_dim=3, latent_dim=256)
        obs = torch.zeros(1, 3, 64, 64)

        a1, _, _ = policy.get_action_and_logp(obs, deterministic=True)
        a2, _, _ = policy.get_action_and_logp(obs, deterministic=True)
        torch.testing.assert_close(a1, a2,
            msg="Deterministic vision policy should return identical actions.")

    def test_stochastic_actions_differ(self):
        """Stochastic sampling should (almost certainly) differ across calls."""
        from visual_rl_locomotion.models.mlp_policy import MLPPolicy
        from visual_rl_locomotion.utils.seed import set_seed

        set_seed(123)
        policy = MLPPolicy(obs_dim=11, action_dim=3)
        obs = torch.zeros(1, 11)

        actions = [
            policy.get_action_and_logp(obs, deterministic=False)[0]
            for _ in range(10)
        ]
        # At least one pair should differ (std initialised to 1).
        any_diff = any(
            not torch.allclose(actions[i], actions[j])
            for i in range(len(actions))
            for j in range(i + 1, len(actions))
        )
        assert any_diff, "Stochastic policy returned identical samples every time."


# ---------------------------------------------------------------------------
# GAE determinism
# ---------------------------------------------------------------------------

class TestGAEDeterminism:
    """compute_gae is a pure function — same inputs always produce same outputs."""

    def test_gae_output_stable(self):
        from visual_rl_locomotion.algo.ppo import compute_gae

        rewards = np.array([1.0, 0.5, -0.2, 1.0], dtype=np.float32)
        values  = np.array([0.8, 0.6,  0.1, 0.3], dtype=np.float32)
        dones   = np.array([0.0, 0.0,  1.0, 0.0], dtype=np.float32)

        adv1, ret1 = compute_gae(rewards, values, dones,
                                 last_value=0.5, gamma=0.99, gae_lambda=0.95)
        adv2, ret2 = compute_gae(rewards, values, dones,
                                 last_value=0.5, gamma=0.99, gae_lambda=0.95)

        np.testing.assert_array_equal(adv1, adv2)
        np.testing.assert_array_equal(ret1, ret2)

    def test_gae_episode_boundary(self):
        """Done=1 at step t must zero out bootstrap from step t+1."""
        from visual_rl_locomotion.algo.ppo import compute_gae

        rewards = np.array([1.0, 1.0], dtype=np.float32)
        values  = np.array([0.5, 0.5], dtype=np.float32)
        dones   = np.array([1.0, 0.0], dtype=np.float32)   # episode ends at step 0

        adv, ret = compute_gae(rewards, values, dones,
                               last_value=10.0, gamma=0.99, gae_lambda=1.0)

        # At step 0: done=1 → next_val should be masked to 0, not 0.5
        # delta_0 = reward_0 + 0.99 * 0.0 * (1-1) - value_0 = 1.0 - 0.5 = 0.5
        expected_adv0 = 1.0 - 0.5   # = 0.5
        assert abs(adv[0] - expected_adv0) < 1e-5, \
            f"GAE did not respect done boundary: adv[0]={adv[0]:.6f}, expected {expected_adv0:.6f}"


# ---------------------------------------------------------------------------
# Seed utility
# ---------------------------------------------------------------------------

class TestSetSeed:
    """set_seed() pins all relevant RNG sources."""

    def test_numpy_reproducible(self):
        from visual_rl_locomotion.utils.seed import set_seed
        set_seed(0)
        a = np.random.randn(10)
        set_seed(0)
        b = np.random.randn(10)
        np.testing.assert_array_equal(a, b)

    def test_torch_reproducible(self):
        from visual_rl_locomotion.utils.seed import set_seed
        set_seed(0)
        a = torch.randn(10)
        set_seed(0)
        b = torch.randn(10)
        torch.testing.assert_close(a, b)


# ---------------------------------------------------------------------------
# Regression test: short training run (marked slow — skipped in CI fast mode)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestShortTrainingRegression:
    """
    Train for a very short budget and verify loss decreases.

    Marked slow — not run in the default CI job.
    Run manually:  pytest tests/ -m slow -v
    """

    def test_value_loss_decreases_state(self):
        """After a handful of updates on a synthetic buffer, value loss should
        change (verifying gradients flow)."""
        from visual_rl_locomotion.algo.ppo import PPOAgent, compute_gae

        obs_dim    = 11
        action_dim = 3
        T          = 256

        agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, lr=3e-4)

        # Constant synthetic reward — value net should learn to predict it.
        buffer = {
            "obs":     np.ones((T, obs_dim), dtype=np.float32),
            "actions": np.zeros((T, action_dim), dtype=np.float32),
            "logps":   np.zeros(T, dtype=np.float32),
            "rewards": np.ones(T, dtype=np.float32),
            "dones":   np.zeros(T, dtype=np.float32),
            "values":  np.zeros(T, dtype=np.float32),
        }
        adv, ret = compute_gae(
            buffer["rewards"], buffer["values"], buffer["dones"],
            last_value=0.0, gamma=0.99, gae_lambda=0.95,
        )
        buffer["advantages"] = adv
        buffer["returns"]    = ret

        m1 = agent.update(buffer, epochs=1,  batch_size=32,
                          clip_range=0.2, vf_coef=0.5,
                          ent_coef=0.0, max_grad_norm=0.5)
        m2 = agent.update(buffer, epochs=10, batch_size=32,
                          clip_range=0.2, vf_coef=0.5,
                          ent_coef=0.0, max_grad_norm=0.5)

        # Value loss should strictly decrease when learning on constant reward.
        assert m2["value_loss"] < m1["value_loss"], \
            f"Value loss did not decrease: {m1['value_loss']:.4f} → {m2['value_loss']:.4f}"

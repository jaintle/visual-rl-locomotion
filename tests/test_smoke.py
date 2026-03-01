"""
Smoke tests — fast environment and model construction checks.

These tests do NOT run full training loops.  They verify:
  - Environment creation succeeds in both observation modes.
  - Observation shapes match declared spaces.
  - Policy / value networks accept the correct input shapes.
  - CNN encoder produces the expected latent shape.
  - A single rollout step executes without errors.

Runtime target: < 60 seconds on CPU.
"""

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Environment smoke tests
# ---------------------------------------------------------------------------

class TestMakeEnv:
    """make_env() constructs and steps correctly in both modes."""

    def test_state_mode_obs_shape(self):
        from visual_rl_locomotion.envs.make_env import make_env
        env = make_env("Hopper-v4", seed=0, obs_mode="state")
        obs, _ = env.reset(seed=0)
        assert obs.ndim == 1, "State obs must be 1-D."
        assert obs.shape == env.observation_space.shape
        env.close()

    def test_state_mode_step(self):
        from visual_rl_locomotion.envs.make_env import make_env
        env = make_env("Hopper-v4", seed=0, obs_mode="state")
        env.reset(seed=0)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        assert obs.shape == env.observation_space.shape
        assert isinstance(float(reward), float)
        env.close()

    def test_pixel_mode_obs_shape(self):
        from visual_rl_locomotion.envs.make_env import make_env
        env = make_env("Hopper-v4", seed=0, obs_mode="pixels", img_size=64)
        obs, _ = env.reset(seed=0)
        assert obs.shape == (3, 64, 64), f"Expected (3,64,64), got {obs.shape}"
        assert obs.dtype == np.float32
        env.close()

    def test_pixel_mode_value_range(self):
        from visual_rl_locomotion.envs.make_env import make_env
        env = make_env("Hopper-v4", seed=0, obs_mode="pixels", img_size=64)
        obs, _ = env.reset(seed=0)
        assert obs.min() >= 0.0, "Pixel values must be ≥ 0."
        assert obs.max() <= 1.0, "Pixel values must be ≤ 1."
        env.close()

    def test_pixel_mode_step(self):
        from visual_rl_locomotion.envs.make_env import make_env
        env = make_env("Hopper-v4", seed=0, obs_mode="pixels", img_size=64)
        env.reset(seed=0)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        assert obs.shape == (3, 64, 64)
        env.close()

    def test_pixel_mode_img_size_32(self):
        from visual_rl_locomotion.envs.make_env import make_env
        env = make_env("Hopper-v4", seed=0, obs_mode="pixels", img_size=32)
        obs, _ = env.reset(seed=0)
        assert obs.shape == (3, 32, 32), f"Expected (3,32,32), got {obs.shape}"
        env.close()


# ---------------------------------------------------------------------------
# Model construction smoke tests
# ---------------------------------------------------------------------------

class TestMLPModels:
    """MLPPolicy and MLPValueNet accept correct shapes."""

    def test_policy_forward(self):
        from visual_rl_locomotion.models.mlp_policy import MLPPolicy
        policy = MLPPolicy(obs_dim=11, action_dim=3)
        obs = torch.zeros(4, 11)
        dist = policy(obs)
        action = dist.sample()
        assert action.shape == (4, 3)

    def test_policy_get_action_stochastic(self):
        from visual_rl_locomotion.models.mlp_policy import MLPPolicy
        policy = MLPPolicy(obs_dim=11, action_dim=3)
        obs = torch.zeros(1, 11)
        action, logp, entropy = policy.get_action_and_logp(obs, deterministic=False)
        assert action.shape == (1, 3)
        assert logp.shape == (1,)
        assert entropy.shape == (1,)

    def test_policy_get_action_deterministic(self):
        from visual_rl_locomotion.models.mlp_policy import MLPPolicy
        policy = MLPPolicy(obs_dim=11, action_dim=3)
        obs = torch.zeros(1, 11)
        action, logp, entropy = policy.get_action_and_logp(obs, deterministic=True)
        assert action.shape == (1, 3)

    def test_policy_evaluate_actions(self):
        from visual_rl_locomotion.models.mlp_policy import MLPPolicy
        policy = MLPPolicy(obs_dim=11, action_dim=3)
        obs     = torch.zeros(8, 11)
        actions = torch.zeros(8, 3)
        logp, entropy = policy.evaluate_actions(obs, actions)
        assert logp.shape    == (8,)
        assert entropy.shape == (8,)

    def test_value_net_forward(self):
        from visual_rl_locomotion.models.mlp_policy import MLPValueNet
        value_net = MLPValueNet(obs_dim=11)
        obs = torch.zeros(4, 11)
        values = value_net(obs)
        assert values.shape == (4,)


class TestCNNEncoder:
    """CNNEncoder produces correct latent shape for various resolutions."""

    @pytest.mark.parametrize("img_size,latent_dim", [
        (64, 256),
        (84, 256),
        (64, 128),
    ])
    def test_encoder_output_shape(self, img_size, latent_dim):
        from visual_rl_locomotion.models.cnn_encoder import CNNEncoder
        enc = CNNEncoder(obs_shape=(3, img_size, img_size), latent_dim=latent_dim)
        x   = torch.zeros(2, 3, img_size, img_size)
        out = enc(x)
        assert out.shape == (2, latent_dim), \
            f"Expected (2, {latent_dim}), got {out.shape}"


class TestVisionModels:
    """VisionPolicy and VisionValueNet accept CHW pixel inputs."""

    OBS_SHAPE  = (3, 64, 64)
    ACTION_DIM = 3
    LATENT_DIM = 256

    def test_vision_policy_get_action(self):
        from visual_rl_locomotion.models.vision_policy import VisionPolicy
        policy = VisionPolicy(self.OBS_SHAPE, self.ACTION_DIM, self.LATENT_DIM)
        obs    = torch.zeros(1, *self.OBS_SHAPE)
        action, logp, entropy = policy.get_action_and_logp(obs, deterministic=False)
        assert action.shape  == (1, self.ACTION_DIM)
        assert logp.shape    == (1,)
        assert entropy.shape == (1,)

    def test_vision_policy_deterministic(self):
        from visual_rl_locomotion.models.vision_policy import VisionPolicy
        policy  = VisionPolicy(self.OBS_SHAPE, self.ACTION_DIM, self.LATENT_DIM)
        obs     = torch.zeros(1, *self.OBS_SHAPE)
        a1, _, _ = policy.get_action_and_logp(obs, deterministic=True)
        a2, _, _ = policy.get_action_and_logp(obs, deterministic=True)
        assert torch.allclose(a1, a2), "Deterministic actions must be identical."

    def test_vision_policy_evaluate_actions(self):
        from visual_rl_locomotion.models.vision_policy import VisionPolicy
        policy  = VisionPolicy(self.OBS_SHAPE, self.ACTION_DIM, self.LATENT_DIM)
        obs     = torch.zeros(4, *self.OBS_SHAPE)
        actions = torch.zeros(4, self.ACTION_DIM)
        logp, entropy = policy.evaluate_actions(obs, actions)
        assert logp.shape    == (4,)
        assert entropy.shape == (4,)

    def test_vision_value_net_forward(self):
        from visual_rl_locomotion.models.vision_policy import VisionValueNet
        vnet   = VisionValueNet(self.OBS_SHAPE, self.LATENT_DIM)
        obs    = torch.zeros(4, *self.OBS_SHAPE)
        values = vnet(obs)
        assert values.shape == (4,)


# ---------------------------------------------------------------------------
# PPO agent smoke test  (1 rollout, 1 update — no gym interaction)
# ---------------------------------------------------------------------------

class TestPPOAgentSmoke:
    """PPOAgent can be constructed and its update loop runs without NaN."""

    def test_compute_gae_basic(self):
        from visual_rl_locomotion.algo.ppo import compute_gae
        T = 32
        rewards = np.ones(T, dtype=np.float32)
        values  = np.zeros(T, dtype=np.float32)
        dones   = np.zeros(T, dtype=np.float32)
        adv, ret = compute_gae(rewards, values, dones,
                               last_value=0.0, gamma=0.99, gae_lambda=0.95)
        assert adv.shape == (T,)
        assert ret.shape == (T,)
        assert not np.any(np.isnan(adv)), "GAE advantages contain NaN."

    def test_ppo_agent_update_no_nan(self):
        from visual_rl_locomotion.algo.ppo import PPOAgent, compute_gae
        T         = 64
        obs_dim   = 11
        action_dim = 3

        agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, lr=3e-4)

        # Synthetic buffer
        buffer = {
            "obs":     np.random.randn(T, obs_dim).astype(np.float32),
            "actions": np.random.randn(T, action_dim).astype(np.float32),
            "logps":   np.random.randn(T).astype(np.float32),
            "rewards": np.random.randn(T).astype(np.float32),
            "dones":   np.zeros(T, dtype=np.float32),
            "values":  np.random.randn(T).astype(np.float32),
        }
        adv, ret = compute_gae(
            buffer["rewards"], buffer["values"], buffer["dones"],
            last_value=0.0, gamma=0.99, gae_lambda=0.95,
        )
        buffer["advantages"] = adv
        buffer["returns"]    = ret

        metrics = agent.update(
            buffer, epochs=2, batch_size=16,
            clip_range=0.2, vf_coef=0.5, ent_coef=0.0, max_grad_norm=0.5,
        )
        for k, v in metrics.items():
            assert not np.isnan(v), f"metric '{k}' is NaN after update."

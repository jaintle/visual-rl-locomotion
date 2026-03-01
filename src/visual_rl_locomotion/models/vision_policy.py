"""
Vision-based policy and value networks for pixel-mode PPO.

Design principles:
  - VisionPolicy and VisionValueNet deliberately use *separate* CNN encoders.
    This avoids shared-state complexity and lets the optimiser treat each
    network independently, matching the MLP baseline structure exactly.
  - Both expose the exact same method signatures as MLPPolicy / MLPValueNet,
    so PPOAgent.collect_rollout() and PPOAgent.update() work unchanged.
  - VisionPPOAgent subclasses PPOAgent and overrides only __init__,
    inheriting all rollout / update / checkpoint logic for free.

Phase 3 restriction: no frame stacking.  Each observation is a single
float32 CHW frame in [0, 1].
"""

import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from visual_rl_locomotion.models.cnn_encoder import CNNEncoder


# ---------------------------------------------------------------------------
# Vision policy network
# ---------------------------------------------------------------------------

class VisionPolicy(nn.Module):
    """
    Diagonal-Gaussian policy for pixel observations.

    Encodes a pixel frame with a CNN, then maps the latent vector to an
    action-distribution mean.  The log_std is a learnable parameter vector
    independent of the observation (same design as MLPPolicy).

    Exposes the same interface as MLPPolicy:
        get_action_and_logp(obs, deterministic) -> (action, logp, entropy)
        evaluate_actions(obs, actions)           -> (logp, entropy)

    Args:
        obs_shape:  CHW observation shape, e.g. (3, 64, 64).
        action_dim: Dimensionality of the continuous action space.
        latent_dim: CNN encoder output size.  Default 256.
    """

    def __init__(
        self,
        obs_shape: Tuple[int, int, int],
        action_dim: int,
        latent_dim: int = 256,
    ) -> None:
        super().__init__()

        self.encoder = CNNEncoder(obs_shape, latent_dim)

        # Policy head: latent -> action mean
        self.policy_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
        )

        # Learnable log_std, initialised to 0 (std = 1).
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs: torch.Tensor) -> Normal:
        """
        Compute the action distribution from a pixel observation batch.

        Args:
            obs: Float32 tensor (B, C, H, W) in [0, 1].

        Returns:
            torch.distributions.Normal with batch shape (B,) and
            event shape (action_dim,).
        """
        latent = self.encoder(obs)               # (B, latent_dim)
        mean   = self.policy_head(latent)        # (B, action_dim)
        std    = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)

    def get_action_and_logp(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample (or select deterministically) an action and its log-prob.

        Args:
            obs:           Float32 tensor (B, C, H, W).
            deterministic: Return distribution mean if True (evaluation mode).

        Returns:
            action  : (B, action_dim)
            logp    : (B,)  — sum over action dims
            entropy : (B,)  — sum over action dims
        """
        dist = self.forward(obs)
        action = dist.mean if deterministic else dist.rsample()
        logp    = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, logp, entropy

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute log-probabilities and entropy for given (obs, action) pairs.
        Used during the PPO update step.

        Args:
            obs:     Float32 tensor (B, C, H, W).
            actions: Float32 tensor (B, action_dim).

        Returns:
            logp:    (B,) — sum over action dims.
            entropy: (B,) — sum over action dims.
        """
        dist = self.forward(obs)
        logp    = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return logp, entropy


# ---------------------------------------------------------------------------
# Vision value network
# ---------------------------------------------------------------------------

class VisionValueNet(nn.Module):
    """
    State-value function V(s) for pixel observations.

    Uses its own CNN encoder (separate from VisionPolicy) so that both
    networks have independent representations.  Exposes the same interface
    as MLPValueNet: forward(obs) -> (B,) scalar values.

    Args:
        obs_shape:  CHW observation shape, e.g. (3, 64, 64).
        latent_dim: CNN encoder output size.  Default 256.
    """

    def __init__(
        self,
        obs_shape: Tuple[int, int, int],
        latent_dim: int = 256,
    ) -> None:
        super().__init__()

        self.encoder = CNNEncoder(obs_shape, latent_dim)

        # Value head: latent -> scalar
        self.value_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Estimate state value from pixel observation.

        Args:
            obs: Float32 tensor (B, C, H, W) in [0, 1].

        Returns:
            Float32 tensor (B,) — one scalar value per observation.
        """
        latent = self.encoder(obs)          # (B, latent_dim)
        return self.value_head(latent).squeeze(-1)  # (B,)


# ---------------------------------------------------------------------------
# Vision PPO Agent  (subclasses PPOAgent — only __init__ changes)
# ---------------------------------------------------------------------------

class VisionPPOAgent:
    """
    PPO agent for pixel-based continuous control.

    Identical to PPOAgent in every respect except that it instantiates
    VisionPolicy + VisionValueNet instead of MLPPolicy + MLPValueNet.
    All rollout collection, update, and checkpoint methods are re-used
    verbatim from PPOAgent via composition (we carry the same attributes).

    This approach was chosen over inheritance to keep the class explicit
    and avoid MRO surprises.

    Args:
        obs_shape:  CHW observation shape tuple, e.g. (3, 64, 64).
        action_dim: Dimensionality of the continuous action space.
        lr:         Adam learning rate.
        device:     torch.device.
        latent_dim: CNN encoder output size.  Default 256.
    """

    def __init__(
        self,
        obs_shape: Tuple[int, int, int],
        action_dim: int,
        lr: float = 3e-4,
        device: torch.device = torch.device("cpu"),
        latent_dim: int = 256,
    ) -> None:
        self.device = device

        self.policy    = VisionPolicy(obs_shape, action_dim, latent_dim).to(device)
        self.value_net = VisionValueNet(obs_shape, latent_dim).to(device)

        # Single optimiser covering both networks (same pattern as PPOAgent).
        self.optimizer = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.value_net.parameters()),
            lr=lr,
        )

    # ------------------------------------------------------------------
    # Delegate all other behaviour to PPOAgent methods directly.
    # We re-implement them here to keep the file self-contained and
    # avoid any dependency on PPOAgent's internal state.
    # ------------------------------------------------------------------

    def collect_rollout(
        self,
        env,
        n_steps: int,
        obs: np.ndarray,
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, float, List[float], List[int]]:
        """
        Identical to PPOAgent.collect_rollout — works for any obs shape.
        self.policy.get_action_and_logp and self.value_net() both accept
        (B, *obs.shape) tensors regardless of whether obs is flat or CHW.
        """
        obs_buf  = np.zeros((n_steps, *obs.shape), dtype=np.float32)
        act_buf  = np.zeros((n_steps, env.action_space.shape[0]), dtype=np.float32)
        logp_buf = np.zeros(n_steps, dtype=np.float32)
        rew_buf  = np.zeros(n_steps, dtype=np.float32)
        done_buf = np.zeros(n_steps, dtype=np.float32)
        val_buf  = np.zeros(n_steps, dtype=np.float32)

        episode_returns: List[float] = []
        episode_lengths: List[int]   = []
        ep_return = 0.0
        ep_length = 0

        self.policy.eval()
        self.value_net.eval()

        for t in range(n_steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32,
                                    device=self.device).unsqueeze(0)

            with torch.no_grad():
                action, logp, _ = self.policy.get_action_and_logp(
                    obs_t, deterministic=False
                )
                value = self.value_net(obs_t)

            action_np = action.squeeze(0).cpu().numpy()

            next_obs, reward, terminated, truncated, _ = env.step(action_np)
            done = bool(terminated or truncated)

            obs_buf[t]  = obs
            act_buf[t]  = action_np
            logp_buf[t] = logp.item()
            rew_buf[t]  = float(reward)
            done_buf[t] = float(done)
            val_buf[t]  = value.item()

            ep_return += float(reward)
            ep_length += 1

            if done:
                episode_returns.append(ep_return)
                episode_lengths.append(ep_length)
                ep_return = 0.0
                ep_length = 0
                obs, _ = env.reset()
            else:
                obs = next_obs

        obs_t = torch.as_tensor(obs, dtype=torch.float32,
                                 device=self.device).unsqueeze(0)
        with torch.no_grad():
            last_value = self.value_net(obs_t).item()

        buffer = {
            "obs":     obs_buf,
            "actions": act_buf,
            "logps":   logp_buf,
            "rewards": rew_buf,
            "dones":   done_buf,
            "values":  val_buf,
        }
        return buffer, obs, last_value, episode_returns, episode_lengths

    def update(
        self,
        buffer: Dict[str, np.ndarray],
        epochs: int,
        batch_size: int,
        clip_range: float,
        vf_coef: float,
        ent_coef: float,
        max_grad_norm: float,
    ) -> Dict[str, float]:
        """PPO update — identical logic to PPOAgent.update."""
        self.policy.train()
        self.value_net.train()

        obs_t    = torch.as_tensor(buffer["obs"],        dtype=torch.float32, device=self.device)
        act_t    = torch.as_tensor(buffer["actions"],    dtype=torch.float32, device=self.device)
        old_lp_t = torch.as_tensor(buffer["logps"],      dtype=torch.float32, device=self.device)
        adv_t    = torch.as_tensor(buffer["advantages"], dtype=torch.float32, device=self.device)
        ret_t    = torch.as_tensor(buffer["returns"],    dtype=torch.float32, device=self.device)

        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        n = obs_t.shape[0]
        metrics: Dict[str, List[float]] = {
            "policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": []
        }

        for _ in range(epochs):
            perm = torch.randperm(n, device=self.device)

            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]

                b_obs    = obs_t[idx]
                b_act    = act_t[idx]
                b_old_lp = old_lp_t[idx]
                b_adv    = adv_t[idx]
                b_ret    = ret_t[idx]

                new_lp, entropy = self.policy.evaluate_actions(b_obs, b_act)

                ratio       = (new_lp - b_old_lp).exp()
                pg_loss1    = -b_adv * ratio
                pg_loss2    = -b_adv * ratio.clamp(1.0 - clip_range, 1.0 + clip_range)
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                values     = self.value_net(b_obs)
                value_loss = F.mse_loss(values, b_ret)

                mean_entropy = entropy.mean()
                loss = policy_loss + vf_coef * value_loss - ent_coef * mean_entropy

                with torch.no_grad():
                    approx_kl = (b_old_lp - new_lp).mean().item()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.value_net.parameters()),
                    max_grad_norm,
                )
                self.optimizer.step()

                metrics["policy_loss"].append(policy_loss.item())
                metrics["value_loss"].append(value_loss.item())
                metrics["entropy"].append(mean_entropy.item())
                metrics["approx_kl"].append(approx_kl)

        return {k: float(np.mean(v)) for k, v in metrics.items()}

    def save_checkpoint(self, path: str, global_step: int) -> None:
        """Identical to PPOAgent.save_checkpoint."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {
                "global_step":     global_step,
                "policy_state":    self.policy.state_dict(),
                "value_net_state": self.value_net.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
            },
            path,
        )
        print(f"[checkpoint] Saved → {path}  (step {global_step})")

    def load_checkpoint(self, path: str) -> int:
        """Identical to PPOAgent.load_checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy_state"])
        self.value_net.load_state_dict(ckpt["value_net_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        print(f"[checkpoint] Loaded ← {path}  (step {ckpt['global_step']})")
        return ckpt["global_step"]

"""
Minimal Proximal Policy Optimisation (PPO) for continuous control.

Implements:
    - On-policy rollout collection (single environment)
    - Generalised Advantage Estimation (GAE-lambda)
    - Clipped surrogate objective
    - Value function loss (MSE)
    - Entropy bonus
    - Gradient clipping
    - Checkpoint save / load

This module intentionally contains no pixel-specific logic.
Pixel support is deferred to Phase 3.

References:
    Schulman et al. (2017) — Proximal Policy Optimization Algorithms
    https://arxiv.org/abs/1707.06347
"""

import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from visual_rl_locomotion.models.mlp_policy import MLPPolicy, MLPValueNet


# ---------------------------------------------------------------------------
# GAE helper (pure NumPy — no autograd needed here)
# ---------------------------------------------------------------------------

def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generalised Advantage Estimation.

    Args:
        rewards:    (T,) reward at each timestep.
        values:     (T,) V(s_t) from the value network.
        dones:      (T,) 1.0 if episode ended at step t, else 0.0.
        last_value: V(s_{T}) bootstrap value for the step after the buffer.
        gamma:      Discount factor.
        gae_lambda: GAE lambda (0 = TD(0), 1 = MC returns).

    Returns:
        advantages: (T,) float32 advantage estimates.
        returns:    (T,) float32 discounted returns (advantages + values).
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0

    for t in reversed(range(T)):
        # Non-terminal mask: 0 at episode boundaries.
        mask = 1.0 - dones[t]
        next_val = last_value if t == T - 1 else values[t + 1]
        # TD error (delta).
        delta = rewards[t] + gamma * next_val * mask - values[t]
        # Recursive GAE accumulation.
        advantages[t] = last_gae = delta + gamma * gae_lambda * mask * last_gae

    returns = advantages + values
    return advantages, returns


# ---------------------------------------------------------------------------
# PPO Agent
# ---------------------------------------------------------------------------

class PPOAgent:
    """
    PPO agent for state-based continuous control.

    Holds the policy and value networks, a shared Adam optimiser,
    and provides:
      - collect_rollout()   : gather on-policy experience
      - update()            : run PPO gradient steps on that experience
      - save_checkpoint()   : persist weights + optimiser state
      - load_checkpoint()   : restore from a checkpoint

    Args:
        obs_dim:       Dimensionality of the observation space.
        action_dim:    Dimensionality of the action space.
        lr:            Learning rate for Adam (shared across policy + value).
        device:        torch.device to run tensors on.
        hidden_sizes:  MLP hidden layer widths for both networks.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        device: torch.device = torch.device("cpu"),
        hidden_sizes: Tuple[int, ...] = (64, 64),
    ) -> None:
        self.device = device

        self.policy = MLPPolicy(obs_dim, action_dim, hidden_sizes).to(device)
        self.value_net = MLPValueNet(obs_dim, hidden_sizes).to(device)

        # Single optimiser over both networks (standard PPO practice).
        self.optimizer = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.value_net.parameters()),
            lr=lr,
        )

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def collect_rollout(
        self,
        env,
        n_steps: int,
        obs: np.ndarray,
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, float, List[float], List[int]]:
        """
        Collect n_steps transitions from the environment using the current policy.

        Args:
            env:     Gymnasium environment (must already be reset externally).
            n_steps: Number of environment steps to collect.
            obs:     Current observation (numpy array) to start from.

        Returns:
            buffer:          Dict with keys obs, actions, logps, rewards,
                             dones, values — each a (n_steps,) or
                             (n_steps, dim) numpy array.
            next_obs:        Observation after the last collected step.
            last_value:      V(next_obs) for GAE bootstrapping.
            episode_returns: List of complete episode returns seen in this rollout.
            episode_lengths: List of corresponding episode lengths.
        """
        obs_buf    = np.zeros((n_steps, *obs.shape), dtype=np.float32)
        act_buf    = np.zeros((n_steps, env.action_space.shape[0]), dtype=np.float32)
        logp_buf   = np.zeros(n_steps, dtype=np.float32)
        rew_buf    = np.zeros(n_steps, dtype=np.float32)
        done_buf   = np.zeros(n_steps, dtype=np.float32)
        val_buf    = np.zeros(n_steps, dtype=np.float32)

        episode_returns: List[float] = []
        episode_lengths: List[int]   = []
        ep_return = 0.0
        ep_length = 0

        self.policy.eval()
        self.value_net.eval()

        for t in range(n_steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

            with torch.no_grad():
                action, logp, _ = self.policy.get_action_and_logp(obs_t, deterministic=False)
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

        # Bootstrap value for the observation after the buffer ends.
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
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

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

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
        """
        Run multiple epochs of mini-batch PPO updates on a collected rollout.

        The buffer must already contain 'advantages' and 'returns' keys
        (added by the training script after compute_gae()).

        Args:
            buffer:        Rollout dict with obs, actions, logps, advantages,
                           returns (all numpy arrays of length n_steps).
            epochs:        Number of passes over the rollout data.
            batch_size:    Mini-batch size (samples per gradient step).
            clip_range:    PPO epsilon — clips the probability ratio.
            vf_coef:       Weight for the value loss term.
            ent_coef:      Weight for the entropy bonus term.
            max_grad_norm: Maximum gradient norm for clipping.

        Returns:
            Dict with mean policy_loss, value_loss, entropy, approx_kl
            averaged over all mini-batch updates in this call.
        """
        self.policy.train()
        self.value_net.train()

        # Move the full buffer to tensors once.
        obs_t     = torch.as_tensor(buffer["obs"],        dtype=torch.float32, device=self.device)
        act_t     = torch.as_tensor(buffer["actions"],    dtype=torch.float32, device=self.device)
        old_lp_t  = torch.as_tensor(buffer["logps"],      dtype=torch.float32, device=self.device)
        adv_t     = torch.as_tensor(buffer["advantages"], dtype=torch.float32, device=self.device)
        ret_t     = torch.as_tensor(buffer["returns"],    dtype=torch.float32, device=self.device)

        # Normalise advantages over the full rollout.
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        n = obs_t.shape[0]
        metrics: Dict[str, List[float]] = {
            "policy_loss": [],
            "value_loss":  [],
            "entropy":     [],
            "approx_kl":   [],
        }

        for _ in range(epochs):
            # Shuffle indices for each epoch.
            perm = torch.randperm(n, device=self.device)

            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]

                b_obs    = obs_t[idx]
                b_act    = act_t[idx]
                b_old_lp = old_lp_t[idx]
                b_adv    = adv_t[idx]
                b_ret    = ret_t[idx]

                # --- Evaluate current policy on batch actions ---
                new_lp, entropy = self.policy.evaluate_actions(b_obs, b_act)

                # --- Policy loss (clipped surrogate) ---
                ratio     = (new_lp - b_old_lp).exp()
                pg_loss1  = -b_adv * ratio
                pg_loss2  = -b_adv * ratio.clamp(1.0 - clip_range, 1.0 + clip_range)
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                # --- Value loss (MSE) ---
                values     = self.value_net(b_obs)
                value_loss = F.mse_loss(values, b_ret)

                # --- Entropy bonus (negative because we maximise entropy) ---
                mean_entropy = entropy.mean()

                # --- Total loss ---
                loss = policy_loss + vf_coef * value_loss - ent_coef * mean_entropy

                # --- Approximate KL (first-order, cheap to compute) ---
                with torch.no_grad():
                    approx_kl = (b_old_lp - new_lp).mean().item()

                # --- Gradient step ---
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

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str, global_step: int) -> None:
        """
        Save policy weights, value-network weights, optimiser state,
        and training step count to a .pt file.

        Args:
            path:        Destination file path (e.g. checkpoints/step_2048.pt).
            global_step: Current environment step count (saved for reference).
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {
                "global_step":      global_step,
                "policy_state":     self.policy.state_dict(),
                "value_net_state":  self.value_net.state_dict(),
                "optimizer_state":  self.optimizer.state_dict(),
            },
            path,
        )
        print(f"[checkpoint] Saved → {path}  (step {global_step})")

    def load_checkpoint(self, path: str) -> int:
        """
        Restore policy, value network, and optimiser from a checkpoint.

        Args:
            path: Path to the .pt file produced by save_checkpoint().

        Returns:
            The global_step stored in the checkpoint.
        """
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy_state"])
        self.value_net.load_state_dict(ckpt["value_net_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        print(f"[checkpoint] Loaded ← {path}  (step {ckpt['global_step']})")
        return ckpt["global_step"]

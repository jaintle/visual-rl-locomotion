"""
MLP actor and critic networks for state-based continuous control.

Architecture:
    - MLPPolicy : obs -> [64, 64] -> action_mean  +  learnable log_std
    - MLPValueNet: obs -> [64, 64] -> scalar V(s)

Both use Tanh activations (standard for MuJoCo locomotion tasks).
The policy outputs a factored Gaussian distribution: each action
dimension is independent with its own mean and a shared-per-dim std.
"""

from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mlp(
    in_dim: int,
    out_dim: int,
    hidden_sizes: Tuple[int, ...],
    activation: type = nn.Tanh,
) -> nn.Sequential:
    """Build a fully-connected network with Tanh activations."""
    layers = []
    prev = in_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(prev, h))
        layers.append(activation())
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Policy network
# ---------------------------------------------------------------------------

class MLPPolicy(nn.Module):
    """
    Diagonal Gaussian policy parameterised by an MLP.

    The mean (mu) is produced by a feed-forward network.
    The log standard deviation is a learnable parameter vector
    (one value per action dimension), independent of the observation.
    This matches the standard PPO setup for continuous control.

    Args:
        obs_dim:      Dimensionality of the observation vector.
        action_dim:   Dimensionality of the action vector.
        hidden_sizes: Tuple of hidden layer widths. Default: (64, 64).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Tuple[int, ...] = (64, 64),
    ) -> None:
        super().__init__()
        self.net = _make_mlp(obs_dim, action_dim, hidden_sizes)
        # Initialise log_std to 0 -> std starts at 1 (moderate exploration).
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs: torch.Tensor) -> Normal:
        """
        Compute the action distribution given an observation batch.

        Args:
            obs: Float tensor of shape (batch, obs_dim).

        Returns:
            A torch.distributions.Normal with batch shape (batch,)
            and event shape (action_dim,).
        """
        mean = self.net(obs)
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)

    def get_action_and_logp(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample (or select) an action and compute its log-probability.

        Args:
            obs:           Float tensor of shape (batch, obs_dim) or (obs_dim,).
            deterministic: If True, return the distribution mean (no sampling).
                           Used during evaluation.

        Returns:
            action : Tensor (batch, action_dim) or (action_dim,).
            logp   : Scalar log-probability of the action, summed over dims.
            entropy: Per-sample entropy, summed over action dims.
        """
        dist = self.forward(obs)
        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()
        logp = dist.log_prob(action).sum(dim=-1)   # (batch,) or scalar
        entropy = dist.entropy().sum(dim=-1)        # (batch,) or scalar
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
            obs:     Float tensor (batch, obs_dim).
            actions: Float tensor (batch, action_dim).

        Returns:
            logp:    Float tensor (batch,) — sum over action dims.
            entropy: Float tensor (batch,) — sum over action dims.
        """
        dist = self.forward(obs)
        logp = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return logp, entropy


# ---------------------------------------------------------------------------
# Value network
# ---------------------------------------------------------------------------

class MLPValueNet(nn.Module):
    """
    State-value function V(s) parameterised by an MLP.

    Args:
        obs_dim:      Dimensionality of the observation vector.
        hidden_sizes: Tuple of hidden layer widths. Default: (64, 64).
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: Tuple[int, ...] = (64, 64),
    ) -> None:
        super().__init__()
        self.net = _make_mlp(obs_dim, 1, hidden_sizes)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Estimate state value.

        Args:
            obs: Float tensor (batch, obs_dim).

        Returns:
            Float tensor (batch,) — one scalar value per observation.
        """
        return self.net(obs).squeeze(-1)

"""Policies for InvertedDoublePendulum: an MLP mean-net, or pymoto's kuramoto_inverted_double_pendulum.

Both are wrapped in GaussianPolicy, which adds a state-independent learnable
log_std and turns the mean-net's raw output into (mean, std) for a diagonal
Gaussian over the 1-dim continuous action. The mean-net itself is the only
thing that differs between policies -- see build_policy() in train.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PolicyNet(nn.Module):
    """Maps a 9-dim InvertedDoublePendulum observation to a scalar action mean."""

    def __init__(self, obs_dim: int = 9, hidden_size: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GaussianPolicy(nn.Module):
    """Wraps a mean-net with a learnable log_std to make a diagonal Gaussian policy.

    The action space is Box(-1, 1); tanh keeps the mean inside those bounds, but
    sampled actions can still land outside them (std is not squashed), so
    callers must clip before stepping the env.
    """

    def __init__(self, mean_net: nn.Module, init_log_std: float = 0.0):
        super().__init__()
        self.mean_net = mean_net
        self.log_std = nn.Parameter(torch.full((1,), init_log_std))

    def get_coupling(self):
        """Delegates to the mean-net; only meaningful when it's a kuramoto model."""
        return self.mean_net.get_coupling()

    @property
    def config(self):
        """Delegates to the mean-net; only meaningful when it's a kuramoto model."""
        return self.mean_net.config

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = torch.tanh(self.mean_net(x))
        std = self.log_std.exp().expand_as(mean)
        return mean, std

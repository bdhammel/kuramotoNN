"""The MLP baseline policy for CartPole.

The Kuramoto-oscillator policy is pymoto's `kuramoto_cartpole`:
create_model("kuramoto_cartpole") builds a KuramotoForClassification with 4
inputs, 64 oscillators and 2 action logits -- the same architecture as the MNIST
classifier, K the only trainable tensor. Empirically, n=64 solves CartPole but
much more slowly and less stably than n=128 (hundreds more episodes, with
visible reward collapses along the way) -- fewer oscillators means less capacity
in K to shape the dynamics, so training leans more on luck.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PolicyNet(nn.Module):
    """Maps a 4-dim CartPole observation to logits over 2 discrete actions."""

    def __init__(self, obs_dim: int = 4, n_actions: int = 2, hidden_size: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

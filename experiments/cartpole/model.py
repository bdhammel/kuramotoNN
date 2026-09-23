"""Policy networks for CartPole: an MLP baseline and a Kuramoto-oscillator policy.

KuramotoPolicy mirrors ../mnist/model.py's KuramotoClassifier: a frozen random
drive projects the observation into n oscillators, a trainable coupling matrix
K evolves their phases under the Kuramoto ODE, and a frozen random head reads
out mean-relative [sin, cos] features into action logits. K is the only
trainable tensor. See mnist/model.py for the derivation of the coupling math
and the calibration procedure this reuses.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


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


def _coupling(theta: Tensor, K_eff: Tensor) -> Tensor:
    """(B, n) phases, (n, n) coupling -> (B, n) coupling velocity.

    sum_j K_ij sin(theta_j - theta_i), factored into two (B,n)@(n,n) matmuls to
    avoid materializing a (B, n, n) phase-difference tensor. See
    mnist/model.py:coupling for the full derivation.
    """
    s, c = torch.sin(theta), torch.cos(theta)
    return c * (s @ K_eff.T) - s * (c @ K_eff.T)


def _euler_rollout(z: Tensor, K_eff: Tensor, g: Tensor | float, h: float, num_steps: int) -> Tensor:
    """(B, n) drive -> (B, n) phases at t = num_steps * h. theta_0 = 0."""
    theta = torch.zeros_like(z)
    for _ in range(num_steps):
        theta = theta + h * (g * z + _coupling(theta, K_eff))
    return theta


class KuramotoPolicy(nn.Module):
    """Frozen random drive -> Kuramoto ODE -> mean-relative sin/cos -> frozen random head.

    n=64 oscillators (half PolicyNet's 128 hidden width) and num_steps=10,
    T=1.0 (mnist/model.py's default). Empirically, n=64 solves CartPole but
    much more slowly and less stably than n=128 (hundreds more episodes, with
    visible reward collapses along the way) -- fewer oscillators means less
    capacity in K to shape the dynamics, so training leans more on luck.
    """

    def __init__(
        self,
        obs_dim: int = 4,
        n_actions: int = 2,
        n: int = 64,
        T: float = 1.0,
        num_steps: int = 10,
        k_scale: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.n = n
        self.T = T
        self.num_steps = num_steps

        # K ~ N(0, 1/n), diagonal zeroed (a null direction of the dynamics; see
        # mnist/model.py:effective_coupling). The only trainable tensor here.
        K = torch.randn(n, n, generator=generator) / math.sqrt(n)
        K = K - torch.diag_embed(K.diagonal())
        self.K = nn.Parameter(K)

        self.register_buffer("W", torch.randn(n, obs_dim, generator=generator))
        self.register_buffer("H", torch.randn(n_actions, 2 * n, generator=generator))
        self.register_buffer("g", torch.tensor(1.0))
        self.register_buffer("tau", torch.tensor(1.0))
        self.register_buffer("k_scale", torch.tensor(float(k_scale)))

    def K_eff(self) -> Tensor:
        return self.k_scale * self.K

    def forward(self, x: Tensor) -> Tensor:
        """(B, obs_dim) -> (B, n_actions) logits."""
        assert x.ndim == 2 and x.shape[1] == self.obs_dim, (
            f"expected (B, {self.obs_dim}), got {tuple(x.shape)}"
        )
        z = x @ self.W.T
        h = self.T / self.num_steps if self.num_steps > 0 else 0.0
        theta = _euler_rollout(z, self.K_eff(), self.g, h, self.num_steps)
        theta = theta - theta.mean(dim=-1, keepdim=True)
        features = torch.cat([torch.sin(theta), torch.cos(theta)], dim=-1)
        return features @ self.H.T / self.tau


@torch.no_grad()
def calibrate(model: KuramotoPolicy, x_cal: Tensor, g: float = 1.0) -> None:
    """Set W's scale, g and tau in place from one batch of visited states.

    Mirrors mnist/model.py:calibrate. Run once before training, on states
    gathered from a random-action rollout (CartPole's observation_space bounds
    are not representative -- see train.py:collect_calibration_batch).
    """
    assert x_cal.ndim == 2 and x_cal.shape[1] == model.obs_dim
    assert torch.isfinite(x_cal).all(), "calibration batch is not finite"

    z = x_cal @ model.W.T
    z_std = z.std()
    assert z_std > 0, "calibration drive has zero variance"
    model.W.div_(z_std)

    model.g.fill_(float(g))

    model.tau.fill_(1.0)
    raw_logits = model(x_cal)
    raw_std = raw_logits.std()
    assert raw_std > 0, "raw logits have zero variance; features may be constant"
    model.tau.fill_(float(raw_std))

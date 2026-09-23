"""The Kuramoto ODE: velocity field plus the module that integrates it over [0, T].

    velocity            dtheta/dt = g * z + coupling(theta)
    KuramotoDynamics    (B, n) drive z -> (B, n) phases theta(T); theta_0 = 0, explicit Euler
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from pymoto.layers.coupling import KuramotoCoupling, coupling
from pymoto.layers.integrators import VectorField, euler_step, rollout, trajectory


def velocity(theta: Tensor, z: Tensor, K_eff: Tensor, g: Tensor | float) -> Tensor:
    """(B, n) -> (B, n). dtheta/dt = g * z + coupling(theta).

    g * z occupies the structural slot of the natural frequency omega in the
    standard Kuramoto equation, dtheta_i/dt = omega_i + sum_j K_ij sin(...), and
    is one in the strict sense: z is computed once before integration and is
    constant in t, so with K = 0 each oscillator would rotate uniformly at rate
    g * z_i.

    The inversion that makes this a classifier is that omega is per-sample rather
    than intrinsic. In the physics omega_i is a fixed property of oscillator i;
    here the input sets it, and that is the only path from x into the dynamics.
    A different digit gives a different frequency spectrum across the n
    oscillators, hence a different synchronization pattern at t = T.

    There is therefore no *additional*, input-independent omega_base: it is
    identically 0. A trainable per-oscillator omega was considered and excluded --
    being identical for every sample it carries no information about x, so it
    could only shift the operating point, and the mean-relative readout removes
    its common component regardless.
    """
    return g * z + coupling(theta, K_eff)


class KuramotoDynamics(nn.Module):
    """(B, n) drive -> (B, n) phases at t = T. The network's encoder.

    Integrates dtheta/dt = g * z + coupling(theta) from theta_0 = 0 with num_steps
    explicit Euler steps of size h = T / num_steps. Every step applies the same
    coupling module, so this is a weight-tied residual stack of depth num_steps.

    theta_0 = 0 exactly. The coupling term vanishes there, so early evolution is
    theta(t) ~= g * z * t and the input reaches the readout only through time.

    Args:
        coupling: the coupling module, built by the caller. Its K is the only
            trainable tensor here.
        T: total integration time.
        num_steps: Euler steps over [0, T]; the depth of the stack.

    Set `grad_checkpointing = True` (or call set_grad_checkpointing on the model)
    to trade one extra forward for O(num_steps) less activation memory in backward.
    """

    def __init__(
        self,
        coupling: KuramotoCoupling,
        T: float = 1.0,
        num_steps: int = 10,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.coupling = coupling
        self.n = coupling.n
        self.T = T
        self.num_steps = num_steps
        self.grad_checkpointing = False

        # g: drive gain in dtheta/dt = g*z + coupling. Denominated in radians --
        # theta_0 = 0 makes the coupling vanish at t = 0, so early evolution is
        # theta(t) ~= g*z*t, and with z at unit variance g *is* the typical phase
        # excursion at t = T. That makes its usable band [0.5, 1.5] interpretable
        # rather than arbitrary: below ~0.3 the phases never leave the region where
        # sin/cos are effectively linear and the nonlinearity buys nothing; above
        # ~1.5 the tails wrap past pi and distinct inputs alias onto the same
        # (sin, cos) pair. The phase_std ~ 1 rad row of the diagnostic table is the
        # direct check on it. Set by calibrate().
        self.register_buffer("g", torch.tensor(1.0, device=device, dtype=dtype))

    @property
    def h(self) -> float:
        """Euler step size T / num_steps (0 when num_steps is 0, where it is unused)."""
        return self.T / self.num_steps if self.num_steps > 0 else 0.0

    def velocity_field(self, z: Tensor) -> VectorField:
        """Bind one batch's constant drive: theta -> dtheta/dt.

        K_eff is computed once here rather than once per step. Hand the result to
        any step in pymoto.layers.integrators.
        """
        K_eff = self.coupling.K_eff()
        return lambda theta: velocity(theta, z, K_eff, self.g)

    def _steps(self, num_steps: int | None) -> tuple[int, float]:
        steps = self.num_steps if num_steps is None else num_steps
        return steps, (self.T / steps if steps > 0 else 0.0)

    def forward(self, z: Tensor, num_steps: int | None = None) -> Tensor:
        """(B, n) -> (B, n). Euler rollout from theta_0 = 0 to t = T.

        `num_steps` may be overridden so an evaluation can run the num_steps = 0
        control; the total integration time T is held fixed either way.
        """
        steps, h = self._steps(num_steps)
        return rollout(
            euler_step, self.velocity_field(z), torch.zeros_like(z), h, steps,
            grad_checkpointing=self.grad_checkpointing,
        )

    def forward_trajectory(self, z: Tensor, num_steps: int | None = None) -> Tensor:
        """(B, n) -> (steps + 1, B, n): theta at t = 0, h, ..., T.

        The per-step intermediates (timm's forward_intermediates, HF's
        hidden_states). The last entry is exactly forward(z, num_steps).
        """
        steps, h = self._steps(num_steps)
        return trajectory(
            euler_step, self.velocity_field(z), torch.zeros_like(z), h, steps,
            grad_checkpointing=self.grad_checkpointing,
        )

    def extra_repr(self) -> str:
        return f"T={self.T:g}, num_steps={self.num_steps}, h={self.h:g}, g={float(self.g):g}"

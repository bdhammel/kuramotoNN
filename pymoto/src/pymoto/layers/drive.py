"""The input stage: a frozen random projection that sets each oscillator's drive."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class FrozenDrive(nn.Module):
    """(B, in_dim) -> (B, n). The constant forcing term z = x @ W.T.

    W is a buffer: drawn N(0, 1) once, rescaled by calibrate_(), never trained.
    Drawn on the CPU in float32, then cast to `device` / `dtype`.
    """

    def __init__(
        self,
        in_dim: int,
        n: int,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.n = n

        # W: (n, in_dim) frozen random projection producing the drive z = x @ W.T.
        # calibrate_() divides it by std(z) measured on one batch, so std(z) == 1.
        #
        # This is a normalization, not a change of units: z is a pure number either
        # way, and theta is in radians regardless, being the argument of sin/cos.
        # What it buys is that g becomes the *only* remaining scale in the drive
        # term, so the number typed for g is itself the phase excursion, via
        # theta(T) ~= g*z*T with std(z) = T = 1.
        #
        # The scale being removed is large and entirely arbitrary. Global-scalar
        # standardization forces E||x||^2 = D, and Var(z_i | x) = ||x||^2, so an
        # unnormalized W ~ N(0, 1) gives std(z) = sqrt(D) = 28 exactly (for MNIST's
        # D = 784) -- meaning g = 1 would drive a 28 rad spread, 4.5 full wraps past
        # 2*pi, everything aliased. It also decouples g from the input dimension:
        # without it, changing D silently re-tunes g by sqrt(D_new / D).
        self.register_buffer("W", torch.empty(n, in_dim, device=device, dtype=dtype))
        self.reset_parameters(generator)

    @torch.no_grad()
    def reset_parameters(self, generator: torch.Generator | None = None) -> None:
        self.W.copy_(torch.randn(self.n, self.in_dim, generator=generator))

    @torch.no_grad()
    def calibrate_(self, x: Tensor) -> None:
        """Rescale W in place so the drive has unit std over the batch `x`."""
        z_std = self(x).std()
        assert z_std > 0, "calibration drive has zero variance"
        self.W.div_(z_std)

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim == 2 and x.shape[1] == self.in_dim, (
            f"expected (B, {self.in_dim}), got {tuple(x.shape)}"
        )
        z = x @ self.W.T
        assert z.shape == (x.shape[0], self.n)
        return z

    def extra_repr(self) -> str:
        return f"in_dim={self.in_dim}, n={self.n}"

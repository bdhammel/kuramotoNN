"""The Kuramoto coupling term and the module that owns K.

    effective_coupling    (n, n) -> (n, n)    k_scale * K
    coupling              (B, n) -> (B, n)    sum_j K_ij sin(theta_j - theta_i), two matmuls
    coupling_pairwise     (B, n) -> (B, n)    the same, literally; reference only
    KuramotoCoupling      nn.Module holding K (the only trainable tensor) and k_scale
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def effective_coupling(K: Tensor, k_scale: Tensor | float) -> Tensor:
    """(n, n) -> (n, n). The coupling matrix as the dynamics see it: k_scale * K.

    K's diagonal is deliberately left in place. An earlier revision returned

        return k_scale * (K - torch.diag_embed(K.diagonal()))

    which was dropped for simplicity, because it cannot change a prediction: the
    j = i term of the coupling is

        K_ii * (cos(theta_i)sin(theta_i) - sin(theta_i)cos(theta_i)) = 0

    identically, so K's diagonal is a null direction of the model. Measured
    difference between a zeroed diagonal and a junk one, over a full batch, is
    1.3e-6 relative -- float32 rounding and nothing else.

    What the subtraction did buy was an exactly-zero *gradient*. Autograd
    evaluates dL/dK as (G*C).T @ S - (G*S).T @ C; on the diagonal those two are
    the same sum in a different multiplication order, so they cancel to ~1e-7 in
    float32 rather than to 0. Adam normalizes by the gradient's own RMS, which
    turns that rounding residue into a near-full-lr step, and over 30 epochs K_ii
    random-walks to roughly the scale of a real off-diagonal entry. Subtracting
    diag_embed(K.diagonal()) made backprop compute G_ii - G_ii on the identical
    float, i.e. bit-exactly zero, pinning K_ii at 0 forever.

    Consequences of dropping it, both handled elsewhere:
      - Trained checkpoints will show a nonzero, slowly growing diagonal. It is
        inert; do not read anything into it.
      - The Jacobian diagnostic is unaffected. J = K_eff - diag(K_eff @ 1) is
        exactly invariant to the diagonal, since adding d to K_ii adds d to both
        the entry and its row sum (verified to 4e-16).
      - The K *norms* are not invariant, and that one matters: a drifted diagonal
        pulls ||K - K.T||_F / ||K||_F down, which is exactly the signature the
        project watches for as evidence of a learned gradient flow. So
        pymoto.diagnostics.compute_diagnostics strips the diagonal before measuring.

    Note the null-direction argument is a property of *this* coupling term, not a
    general fact. Under a Sakaguchi phase lag sin(theta_j - theta_i - phi) the
    j = i term is -sin(phi) != 0 and the diagonal becomes a live self-drive.
    """
    assert K.ndim == 2 and K.shape[0] == K.shape[1], f"K must be square, got {tuple(K.shape)}"
    return k_scale * K


def coupling(theta: Tensor, K_eff: Tensor) -> Tensor:
    """(B, n) phases, (n, n) coupling -> (B, n) coupling velocity.

    This is the standard Kuramoto interaction

        c_i = sum_j K_ij * sin(theta_j - theta_i)

    expanded with sin(a - b) = sin(a)cos(b) - cos(a)sin(b) into

        c_i = cos(theta_i) * sum_j K_ij sin(theta_j)
            - sin(theta_i) * sum_j K_ij cos(theta_j)

    so it costs two (B, n) @ (n, n) matmuls instead of materializing the
    (B, n, n) per-sample phase-difference tensor.

    Measured against coupling_pairwise below, which is the literal form, at
    B = 128, n = 256, num_steps = 10:

        wall clock, 10 steps      91.3 ms  ->        1.1 ms   ( 83x faster)
        retained for backward      960 MB  ->          5 MB   (192x smaller)
        sin/cos evaluations    83,886,080  ->      655,360    (128x fewer)
        multiply-adds          83,886,080  ->  167,772,160    (  2x MORE)

    Three separate effects, and memory is the binding one: the (B, n, n) tensor is
    not transient, because autograd retains every step's intermediates for the
    backward pass and there are num_steps of them -- about 3.8 GB of activations
    at batch 512. Second, transcendentals collapse from B*n^2 to 2*B*n, since
    sin/cos are evaluated once per oscillator rather than once per pair. Third,
    the remaining arithmetic becomes GEMM: this form does twice the multiply-adds
    and is still 83x faster, because a matmul is compute-bound on a heavily tuned
    kernel while an elementwise-plus-reduction over 32 MB is bandwidth-bound.

    The factorization exists because sin(theta_j - theta_i) separates into terms
    that each depend on a single index, which turns sum_j K_ij sin(theta_j) into a
    matrix product. A general f(theta_j - theta_i) would not separate, and the
    (B, n, n) form would be the only option.
    """
    assert theta.ndim == 2 and theta.shape[1] == K_eff.shape[0]
    s, c = torch.sin(theta), torch.cos(theta)
    # (s @ K_eff.T)[b, i] == sum_j K_eff[i, j] * sin(theta[b, j])
    return c * (s @ K_eff.T) - s * (c @ K_eff.T)


def coupling_pairwise(theta: Tensor, K_eff: Tensor) -> Tensor:
    """(B, n) phases, (n, n) coupling -> (B, n). Reference implementation.

    A literal transcription of

        c_i = sum_j K_ij * sin(theta_j - theta_i)

    materializing the (B, n, n) phase-difference tensor. This is the form in
    experiments/sandbox/metronome.py, which is the right choice there: N = 3, a
    single sample, no backward pass, so the difference matrix is nine elements.
    The costs in coupling()'s docstring only appear once there is a batch
    dimension.

    Reference only -- never called on the model path. It exists so the identity
    used by coupling() is checkable rather than asserted:

        assert torch.allclose(coupling(theta, K), coupling_pairwise(theta, K),
                              atol=1e-5)

    (float32 agreement is ~1.5e-6 at these shapes; the two differ only in
    summation order.)
    """
    assert theta.ndim == 2 and theta.shape[1] == K_eff.shape[0]
    # [b, i, j] = theta_bj - theta_bi. Note the transpose convention: unlike
    # metronome.py's (1, N) theta, a batched theta needs both axes named
    # explicitly, and it is that expansion that costs (B, n, n).
    diff = theta[:, None, :] - theta[:, :, None]
    return torch.einsum("ij,bij->bi", K_eff, torch.sin(diff))


class KuramotoCoupling(nn.Module):
    """(B, n) phases -> (B, n) coupling velocity. Owns K, the only trainable tensor.

    K is an nn.Parameter. k_scale is a buffer: it serializes with the checkpoint
    but never receives a gradient.

    Like every pymoto layer, K is drawn on the CPU in float32 from `generator` (the
    global RNG if None) and then cast to `device` / `dtype`, so a seed gives the
    same init regardless of where the model runs.
    """

    def __init__(
        self,
        n: int,
        k_scale: float = 1.0,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.n = n
        self.K = nn.Parameter(torch.empty(n, n, **factory_kwargs))

        # k_scale sets the balance between the two terms of the velocity,
        # dtheta/dt = g*z + k_scale * (coupling), and is read off the rho
        # diagnostic, rho = ||coupling|| / ||g*z||, whose usable band is 0.3 - 2.
        #
        # It selects the regime the ODE runs in. Too small and the oscillators
        # barely interact: theta ~= g*z*T, the readout is a fixed pointwise
        # nonlinearity applied to a random projection, and there is no dynamics
        # for K to shape. Too large and the coupling synchronizes the population,
        # phases collapse toward a common value, the mean-relative readout
        # subtracts most of what is left, and the features rank-collapse -- which
        # shows up immediately as a low participation ratio.
        #
        # Keeping it separate from K's initializer rather than folding it into the
        # init std is what makes a k_scale sweep meaningful: at a fixed seed every
        # value of k_scale gets the *same* random K, varying only its strength.
        self.register_buffer("k_scale", torch.tensor(float(k_scale), **factory_kwargs))

        self.reset_parameters(generator)

    @torch.no_grad()
    def reset_parameters(self, generator: torch.Generator | None = None) -> None:
        # K ~ N(0, 1/n), diagonal zeroed. The only trainable tensor in the model.
        # The diagonal is zeroed here only so runs start from a clean, inspectable
        # state; it is a null direction of the dynamics (see effective_coupling)
        # and will drift away from zero during training. That is expected.
        K = torch.randn(self.n, self.n, generator=generator) / math.sqrt(self.n)
        self.K.copy_(K - torch.diag_embed(K.diagonal()))

    def K_eff(self) -> Tensor:
        """(n, n). The coupling matrix as the dynamics actually see it."""
        return effective_coupling(self.K, self.k_scale)

    def forward(self, theta: Tensor) -> Tensor:
        return coupling(theta, self.K_eff())

    def extra_repr(self) -> str:
        return f"n={self.n}, k_scale={float(self.k_scale):g}"

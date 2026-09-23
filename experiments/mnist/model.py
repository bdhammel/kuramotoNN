"""The Kuramoto classifier: a frozen projection, a coupled-oscillator ODE, a frozen readout.

The coupling matrix K is the only trainable tensor in the model. The input enters
only as a constant drive term in the ODE and the initial phases are exactly zero,
so with zero integration steps the input has no path whatsoever to the output.
Any accuracy above chance is therefore attributable to K and to the dynamics.

Layout:
    effective_coupling -> coupling -> velocity -> euler_rollout / rk4_rollout
    coupling_pairwise   (reference implementation of coupling, never on the model path)
    readout_features
    KuramotoClassifier
    calibrate
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
        utils.compute_diagnostics strips the diagonal before measuring.

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
    metronome.py, which is the right choice there: N = 3, a single sample, no
    backward pass, so the difference matrix is nine elements. The costs in
    coupling()'s docstring only appear once there is a batch dimension.

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


def euler_rollout(
    z: Tensor, K_eff: Tensor, g: Tensor | float, h: float, num_steps: int
) -> Tensor:
    """(B, n) drive -> (B, n) phases at t = num_steps * h.

    Explicit Euler, written out by hand rather than handed to an ODE library, so
    that the structure stays visible: each iteration is a residual update with
    tied weights, i.e. the rollout is a weight-tied ResNet of depth num_steps.

    theta_0 = 0 exactly. The coupling term vanishes there, so early evolution is
    theta(t) ~= g * z * t and the input reaches the readout only through time.
    """
    assert num_steps >= 0
    theta = torch.zeros_like(z)
    for _ in range(num_steps):
        theta = theta + h * velocity(theta, z, K_eff, g)
    return theta


def rk4_rollout(
    z: Tensor, K_eff: Tensor, g: Tensor | float, h: float, num_steps: int
) -> Tensor:
    """(B, n) drive -> (B, n) phases. Classical RK4.

    Evaluation only. The model is always integrated with Euler; this exists so
    eval.py can ask whether the trained K describes a flow or merely a fixed
    discretization. It is not a solver toggle on the model.
    """
    assert num_steps >= 0
    theta = torch.zeros_like(z)
    for _ in range(num_steps):
        k1 = velocity(theta, z, K_eff, g)
        k2 = velocity(theta + 0.5 * h * k1, z, K_eff, g)
        k3 = velocity(theta + 0.5 * h * k2, z, K_eff, g)
        k4 = velocity(theta + h * k3, z, K_eff, g)
        theta = theta + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return theta


def readout_features(theta: Tensor) -> Tensor:
    """(B, n) -> (B, 2n). Mean-relative phases, then [sin, cos] concatenated.

    Subtracting the per-sample mean phase removes the ODE's exact global rotation
    symmetry. Alternative considered: relativizing against a designated reference
    oscillator. Rejected because that oscillator's own trajectory would then leak
    into all 2n features as common-mode noise; the mean is the lower-variance
    estimator of the same quantity.
    """
    assert theta.ndim == 2
    theta = theta - theta.mean(dim=-1, keepdim=True)
    return torch.cat([torch.sin(theta), torch.cos(theta)], dim=-1)


class KuramotoClassifier(nn.Module):
    """Frozen random drive -> Kuramoto ODE -> mean-relative sin/cos -> frozen random head.

    K is an nn.Parameter. W, H, g, tau and k_scale are buffers, so they serialize
    with the checkpoint but never receive gradients.
    """

    def __init__(
        self,
        n: int = 256,
        in_dim: int = 784,
        num_classes: int = 10,
        T: float = 1.0,
        num_steps: int = 10,
        k_scale: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        self.n = n
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.T = T
        self.num_steps = num_steps

        # K ~ N(0, 1/n), diagonal zeroed. The only trainable tensor in the model.
        # The diagonal is zeroed here only so runs start from a clean, inspectable
        # state; it is a null direction of the dynamics (see effective_coupling)
        # and will drift away from zero during training. That is expected.
        K = torch.randn(n, n, generator=generator) / math.sqrt(n)
        K = K - torch.diag_embed(K.diagonal())
        self.K = nn.Parameter(K)

        # W: (n, 784) frozen random projection producing the drive z = x @ W.T.
        # calibrate() divides it by std(z) measured on one batch, so std(z) == 1.
        #
        # This is a normalization, not a change of units: z is a pure number either
        # way, and theta is in radians regardless, being the argument of sin/cos.
        # What it buys is that g becomes the *only* remaining scale in the drive
        # term, so the number typed for g is itself the phase excursion, via
        # theta(T) ~= g*z*T with std(z) = T = 1.
        #
        # The scale being removed is large and entirely arbitrary. Global-scalar
        # standardization forces E||x||^2 = D, and Var(z_i | x) = ||x||^2, so an
        # unnormalized W ~ N(0, 1) gives std(z) = sqrt(D) = 28 exactly -- meaning
        # g = 1 would drive a 28 rad spread, 4.5 full wraps past 2*pi, everything
        # aliased. It also decouples g from the input dimension: without it,
        # changing D silently re-tunes g by sqrt(D_new / 784).
        self.register_buffer("W", torch.randn(n, in_dim, generator=generator))

        # H: (10, 2n) readout head. The final classifier layer, except it is never
        # trained -- a fixed random projection from the 512-dim feature space to 10
        # class scores, drawn before it has seen any data and no bias.
        #
        # This is why the participation ratio is the diagnostic that matters most.
        # A random projection samples the feature covariance roughly
        # democratically; it has no opportunity to seek out the informative
        # directions. A *trained* head could find 10 good directions inside a
        # rank-3 feature set, but H cannot, so the classes have to be separable
        # along generic directions and at least ~10 must be well populated. Control
        # 4, the linear probe, is exactly H made trainable as a measurement.
        #
        # Drawn N(0, 1) and never rescaled: tau absorbs its scale, so normalizing
        # both would be redundant.
        self.register_buffer("H", torch.randn(num_classes, 2 * n, generator=generator))

        # g: drive gain in dtheta/dt = g*z + coupling. Denominated in radians --
        # theta_0 = 0 makes the coupling vanish at t = 0, so early evolution is
        # theta(t) ~= g*z*t, and with z at unit variance g *is* the typical phase
        # excursion at t = T. That makes its usable band [0.5, 1.5] interpretable
        # rather than arbitrary: below ~0.3 the phases never leave the region where
        # sin/cos are effectively linear and the nonlinearity buys nothing; above
        # ~1.5 the tails wrap past pi and distinct inputs alias onto the same
        # (sin, cos) pair. The phase_std ~ 1 rad row of the diagnostic table is the
        # direct check on it.
        self.register_buffer("g", torch.tensor(1.0))

        # tau: logit temperature in logits = features @ H.T / tau. Set by
        # calibrate() to the std of the raw pre-temperature logits, so logits have
        # unit scale at init.
        #
        # It cannot change accuracy: dividing every logit by the same positive
        # constant cannot change an argmax. It only conditions the cross-entropy
        # gradient, which is otherwise at the mercy of however H @ features happens
        # to scale -- raw logits with std ~30 saturate the softmax and the gradient
        # vanishes, std ~0.01 leaves the loss nearly flat at ln(10) = 2.303. Being
        # unable to affect predictions is what makes it safe to fit on data without
        # touching the attribution argument.
        self.register_buffer("tau", torch.tensor(1.0))

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
        self.register_buffer("k_scale", torch.tensor(float(k_scale)))

    @property
    def h(self) -> float:
        """Euler step size T / num_steps (0 when num_steps is 0, where it is unused)."""
        return self.T / self.num_steps if self.num_steps > 0 else 0.0

    def K_eff(self) -> Tensor:
        """(n, n). The coupling matrix as the dynamics actually see it."""
        return effective_coupling(self.K, self.k_scale)

    def drive(self, x: Tensor) -> Tensor:
        """(B, 784) -> (B, n). The constant forcing term z = x @ W.T."""
        assert x.ndim == 2 and x.shape[1] == self.in_dim, (
            f"expected (B, {self.in_dim}), got {tuple(x.shape)}"
        )
        z = x @ self.W.T
        assert z.shape == (x.shape[0], self.n)
        return z

    def integrate(self, z: Tensor, num_steps: int | None = None) -> Tensor:
        """(B, n) -> (B, n). Euler rollout from theta_0 = 0 to t = T.

        `num_steps` may be overridden so eval.py can run the num_steps = 0 control;
        the total integration time T is held fixed either way.
        """
        steps = self.num_steps if num_steps is None else num_steps
        h = self.T / steps if steps > 0 else 0.0
        return euler_rollout(z, self.K_eff(), self.g, h, steps)

    def readout(self, theta: Tensor) -> Tensor:
        """(B, n) -> (B, 2n)."""
        features = readout_features(theta)
        assert features.shape == (theta.shape[0], 2 * self.n)
        return features

    def classify(self, features: Tensor) -> Tensor:
        """(B, 2n) -> (B, num_classes). Frozen random head, no bias."""
        assert features.ndim == 2 and features.shape[1] == 2 * self.n
        return features @ self.H.T / self.tau

    def forward(self, x: Tensor) -> Tensor:
        """(B, 784) -> (B, num_classes) logits."""
        z = self.drive(x)
        z = self.integrate(z)
        z = self.readout(z)
        y = self.classify(z)
        return y


@torch.no_grad()
def calibrate(model: KuramotoClassifier, x_cal: Tensor, g: float = 1.0) -> None:
    """Set W's scale, g and tau in place from one calibration batch. Run once, at init.

    Args:
        model: freshly constructed classifier, modified in place.
        x_cal: (n_cal, 784) standardized calibration batch.
        g: drive gain in radians. Usable range is roughly [0.5, 1.5]; below ~0.3 the
            sin/cos features are effectively linear, above ~1.5 the tails wrap past
            pi and distinct inputs alias onto each other.

    This is deliberately one readable function rather than logic scattered through
    __init__, because it is what makes the model trainable at all.
    """
    assert x_cal.ndim == 2 and x_cal.shape[1] == model.in_dim
    assert torch.isfinite(x_cal).all(), "calibration batch is not finite"

    # Step 2: normalize W empirically so the drive has unit variance by
    # construction. Everything downstream is then denominated in radians.
    z = x_cal @ model.W.T
    z_std = z.std()
    assert z_std > 0, "calibration drive has zero variance"
    model.W.div_(z_std)

    # Step 3: because theta_0 = 0 the coupling term vanishes at t = 0, so
    # theta(t) ~= g * z * t and std_i(theta_i(T)) ~= g. g *is* the typical phase
    # excursion in radians at readout.
    model.g.fill_(float(g))

    # Step 4: tau is the std of the raw pre-temperature logits, so logits have unit
    # scale at init. Dividing every logit by a positive constant cannot change the
    # argmax and therefore cannot change accuracy; it only conditions the
    # cross-entropy gradient.
    model.tau.fill_(1.0)
    raw_logits = model(x_cal)
    raw_std = raw_logits.std()
    assert raw_std > 0, "raw logits have zero variance; features may be constant"
    model.tau.fill_(float(raw_std))

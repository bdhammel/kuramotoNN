"""Seeding, diagnostics, spectral quantities and checkpoint I/O.

The diagnostics here are the cheap early-warning system for the whole project: a
rank-collapsed feature set or an Euler step that cannot resolve the learned
dynamics should be visible in seconds rather than after a training run.

All linear algebra is done on CPU in float64. The matrices are 256 x 256, so this
costs nothing, and `torch.linalg.eigvals` has no MPS kernel.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from model import KuramotoClassifier, coupling


def set_seed(seed: int) -> torch.Generator:
    """Seed python, numpy and torch. Returns a CPU generator for explicit use."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return torch.Generator().manual_seed(seed)


def pick_device(requested: str = "auto") -> torch.device:
    """Resolve 'auto' to cuda, then mps, then cpu."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def participation_ratio(features: Tensor) -> float:
    """(B, d) -> scalar. (sum lam)^2 / sum(lam^2) over eigenvalues of the feature covariance.

    The effective number of well-populated feature directions. It matters more than
    the other diagnostics: a frozen random 2n -> 10 projection samples the feature
    covariance roughly democratically, so at least ~10 populated directions are
    needed to separate 10 classes. Bounded above by min(B - 1, d), which is why the
    calibration batch is larger than the training batch.
    """
    f = features.detach().cpu().to(torch.float64)
    f = f - f.mean(dim=0, keepdim=True)
    # Eigenvalues of the covariance are the squared singular values, up to a
    # constant that cancels in the ratio.
    lam = torch.linalg.svdvals(f) ** 2
    denom = (lam**2).sum()
    if denom <= 0:
        return 0.0
    return float(lam.sum() ** 2 / denom)


def spectral_norm(matrix: Tensor) -> float:
    """(n, n) -> largest singular value."""
    m = matrix.detach().cpu().to(torch.float64)
    return float(torch.linalg.svdvals(m)[0])


def jacobian_spectral_radius(K_eff: Tensor) -> float:
    """(n, n) -> max |eigenvalue| of J = K_eff - diag(K_eff @ 1).

    J is the Jacobian of the coupling term at theta = 0: off the diagonal
    d c_i / d theta_j = K_ij, and on it d c_i / d theta_i = -sum_j K_ij.

    J is not symmetric, so its eigenvalues are complex; the magnitude is what
    bounds the explicit-Euler stability region, hence abs().max().

    Safe to pass K_eff with its diagonal intact: adding d to K_ii adds d to both
    the entry and its row sum, so J is exactly unchanged (verified to 4e-16).
    """
    j = K_eff.detach().cpu().to(torch.float64)
    j = j - torch.diag_embed(j.sum(dim=1))
    return float(torch.linalg.eigvals(j).abs().max())


@dataclass(frozen=True)
class Diagnostics:
    """One calibration batch integrated to t = T, with K at its current value."""

    phase_std: float  # std_i(theta - theta_bar), target ~= 1 rad -> else adjust g
    frac_wrapped: float  # frac(|theta - theta_bar| > pi), target < 1% -> else g too large
    mean_var_ratio: float  # Var(theta_bar) / Var(theta - theta_bar), target << 1
    rho: float  # ||coupling|| / ||g*z||, target 0.3 - 2 -> else adjust k_scale
    participation_ratio: float  # of the 2n features, target >> 10
    h_lambda_max: float  # h * max|lambda(J)|, target << 1 -> else step size unsafe
    k_spectral_norm: float  # ||K_eff||_2, the step-size headroom monitor
    k_frobenius: float  # ||K_eff||_F
    k_asymmetry: float  # ||K - K.T||_F / ||K||_F, -> 0 means a gradient flow

    def as_dict(self, prefix: str = "") -> dict[str, float]:
        return {f"{prefix}{k}": v for k, v in asdict(self).items()}


# (attribute, label, target string) in the order the spec's table lists them.
_DIAGNOSTIC_ROWS: tuple[tuple[str, str, str], ...] = (
    ("phase_std", "std_i(theta - theta_bar)", "~ 1 rad"),
    ("frac_wrapped", "frac(|theta - theta_bar| > pi)", "< 1%"),
    ("mean_var_ratio", "Var(theta_bar)/Var(theta-theta_bar)", "<< 1"),
    ("rho", "rho = ||coupling|| / ||g*z||", "0.3 - 2"),
    ("participation_ratio", "participation ratio (2n feats)", ">> 10"),
    ("h_lambda_max", "h * lambda_max(J)", "<< 1"),
    ("k_spectral_norm", "||K_eff||_2", "-"),
    ("k_frobenius", "||K_eff||_F", "-"),
    ("k_asymmetry", "||K - K.T||_F / ||K||_F", "-"),
)


@torch.no_grad()
def compute_diagnostics(model: KuramotoClassifier, x_cal: Tensor) -> Diagnostics:
    """Run one calibration batch to t = T and measure everything worth watching."""
    z = model.drive(x_cal)
    theta = model.integrate(z)
    theta_rel = theta - theta.mean(dim=-1, keepdim=True)
    theta_bar = theta.mean(dim=-1)
    features = model.readout(theta)

    K_eff = model.K_eff()
    coupling_velocity = coupling(theta, K_eff)
    drive_velocity = model.g * z

    # The norms are measured with the diagonal stripped. The ODE ignores K's
    # diagonal entirely (model.effective_coupling), but it is no longer held at
    # zero, and Adam walks it out to roughly the scale of a real entry over a
    # training run. Left in, those inert entries drag ||K - K.T||_F / ||K||_F
    # down -- which is precisely the signature read as evidence that the model
    # learned a gradient flow. Measuring what the dynamics actually see costs one
    # subtraction on a 256x256 matrix, off the hot path.
    K_meas = K_eff - torch.diag_embed(K_eff.diagonal())
    k_fro = torch.linalg.matrix_norm(K_meas, ord="fro")
    asym = torch.linalg.matrix_norm(K_meas - K_meas.T, ord="fro") / k_fro.clamp_min(1e-12)

    return Diagnostics(
        # std over oscillators i, then averaged over the batch.
        phase_std=float(theta_rel.std(dim=-1).mean()),
        frac_wrapped=float((theta_rel.abs() > math.pi).float().mean()),
        # theta_bar varies across the batch; theta_rel varies within each sample.
        mean_var_ratio=float(theta_bar.var() / theta_rel.var().clamp_min(1e-12)),
        rho=float(coupling_velocity.norm() / drive_velocity.norm().clamp_min(1e-12)),
        participation_ratio=participation_ratio(features),
        # K_eff, not K_meas: the Jacobian is exactly invariant to the diagonal.
        h_lambda_max=model.h * jacobian_spectral_radius(K_eff),
        k_spectral_norm=spectral_norm(K_meas),
        k_frobenius=float(k_fro),
        k_asymmetry=float(asym),
    )


def format_diagnostics(
    diag: Diagnostics,
    title: str = "Init diagnostics",
    extra: dict[str, float] | None = None,
) -> str:
    """Render the diagnostic table for stdout.

    `extra` appends untargeted rows (the calibrated scalars g, tau, k_scale) in the
    same column layout.
    """
    values = asdict(diag)
    rows = [(label, values[key], target) for key, label, target in _DIAGNOSTIC_ROWS]
    rows += [(label, value, "-") for label, value in (extra or {}).items()]

    width = max(len(label) for label, _, _ in rows)
    rule = "-" * (width + 26)
    lines = [title, rule, f"{'quantity'.ljust(width)}  {'value'.rjust(12)}  target", rule]
    lines += [f"{label.ljust(width)}  {value:12.4g}  {target}" for label, value, target in rows]
    lines.append(rule)
    return "\n".join(lines)


def warn_on_diagnostics(diag: Diagnostics) -> list[str]:
    """Return human-readable warnings for out-of-range diagnostics (empty if healthy)."""
    warnings: list[str] = []
    if not 0.5 <= diag.phase_std <= 1.5:
        warnings.append(
            f"phase_std {diag.phase_std:.3g} rad outside [0.5, 1.5]: adjust --g."
        )
    if diag.frac_wrapped > 0.01:
        warnings.append(
            f"{diag.frac_wrapped:.2%} of phases wrapped past pi: --g too large, inputs alias."
        )
    if diag.mean_var_ratio > 0.1:
        warnings.append(
            f"Var(theta_bar)/Var(theta_rel) = {diag.mean_var_ratio:.3g}: standardization is broken."
        )
    if not 0.3 <= diag.rho <= 2.0:
        warnings.append(f"rho {diag.rho:.3g} outside [0.3, 2]: adjust --k-scale.")
    if diag.participation_ratio < 10.0:
        warnings.append(
            f"participation ratio {diag.participation_ratio:.3g} < 10: features are "
            "rank-collapsed and no amount of training K will separate 10 classes."
        )
    if diag.h_lambda_max > 0.5:
        warnings.append(
            f"h*lambda_max(J) = {diag.h_lambda_max:.3g}: Euler step size is unsafe."
        )
    return warnings


def save_checkpoint(
    path: str,
    model: KuramotoClassifier,
    hparams: dict[str, Any],
    K_init: Tensor,
    data_stats: dict[str, float],
    extra: dict[str, Any] | None = None,
) -> None:
    """Write a checkpoint.

    K_init is stored alongside the trained state because the random-K control in
    eval.py needs the exact initial coupling matrix, not a freshly reseeded one.
    """
    torch.save(
        {
            "state_dict": model.state_dict(),
            "K_init": K_init.detach().cpu().clone(),
            "hparams": hparams,
            "data_stats": data_stats,
            "extra": extra or {},
        },
        path,
    )


def load_checkpoint(
    path: str, device: torch.device
) -> tuple[KuramotoClassifier, dict[str, Any]]:
    """Rebuild the model from a checkpoint. Returns (model on device, raw checkpoint)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    hp = ckpt["hparams"]
    model = KuramotoClassifier(
        n=hp["n"],
        T=hp["T"],
        num_steps=hp["num_steps"],
        k_scale=hp["k_scale"],
    )
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device), ckpt


def assert_only_K_is_trainable(model: KuramotoClassifier) -> None:
    """Hard guard on the project's central claim.

    If this ever fires, every reported accuracy has lost its attribution to K.
    """
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    assert trainable == ["K"], (
        f"K must be the only trainable tensor; found {trainable}. "
        "Adding trainable parameters invalidates the entire attribution argument."
    )
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_trainable == model.n * model.n, (
        f"expected {model.n * model.n} trainable scalars, found {n_trainable}"
    )

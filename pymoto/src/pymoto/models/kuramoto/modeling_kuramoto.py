"""The Kuramoto network: a frozen projection, a coupled-oscillator ODE, a frozen readout.

The coupling matrix K is the only trainable tensor in the model. The input enters
only as a constant drive term in the ODE and the initial phases are exactly zero,
so with zero integration steps the input has no path whatsoever to the output.
Any accuracy above chance is therefore attributable to K and to the dynamics.

    KuramotoForClassification
    ├── kuramoto: KuramotoModel
    │   ├── drive:    FrozenDrive          x (B, in_dim) -> z (B, n)          W frozen
    │   ├── dynamics: KuramotoDynamics     z -> theta(T) (B, n)               Euler, theta_0 = 0
    │   │   └── coupling: KuramotoCoupling                                    K trainable
    │   └── readout:  PhaseReadout         theta -> features (B, 2n)
    └── head: FrozenHead                   features -> logits (B, C)         H frozen

Layout of this file, following Hugging Face's modeling_<name>.py:
    KuramotoModelOutput            what the base model returns
    KuramotoPreTrainedModel        config_class / base_model_prefix
    KuramotoModel                  base model: everything but the head
    KuramotoForClassification      base model + head; timm's forward_features / forward_head
    calibrate                      fit W's scale, g and tau on one batch
    checkpoint_filter_fn           load pre-pymoto flat-key checkpoints
    kuramoto_mnist, ...            registered variants (timm's create_model entrypoints)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from pymoto.layers import FrozenDrive, FrozenHead, KuramotoCoupling, KuramotoDynamics, PhaseReadout
from pymoto.modeling_utils import PreTrainedModel
from pymoto.models._registry import register_model
from pymoto.models.kuramoto.configuration_kuramoto import KuramotoConfig


@dataclass
class KuramotoModelOutput:
    """Every intermediate of one forward pass, in order.

    Attributes:
        drive: (B, n) z = x @ W.T, the constant forcing term.
        phases: (B, n) theta at t = T.
        features: (B, 2n) mean-relative [sin, cos] of the phases.
        trajectory: (num_steps + 1, B, n) theta at t = 0, h, ..., T, when requested
            with output_trajectory=True (HF's output_hidden_states); else None.
    """

    drive: Tensor
    phases: Tensor
    features: Tensor
    trajectory: Tensor | None = None


class KuramotoPreTrainedModel(PreTrainedModel):
    config_class = KuramotoConfig
    base_model_prefix = "kuramoto"


class KuramotoModel(KuramotoPreTrainedModel):
    """(B, in_dim) -> KuramotoModelOutput. Drive -> dynamics -> readout, no head."""

    def __init__(
        self,
        config: KuramotoConfig,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(config)
        factory_kwargs = {"device": device, "dtype": dtype}
        # Draw order K, then W (then H, in the task head) is the order the
        # pre-pymoto KuramotoClassifier and KuramotoPolicy used, so a given seed
        # reproduces their init bit-for-bit. Hence coupling is built first.
        coupling = KuramotoCoupling(config.n, config.k_scale, generator=generator, **factory_kwargs)
        self.drive = FrozenDrive(config.in_dim, config.n, generator=generator, **factory_kwargs)
        self.dynamics = KuramotoDynamics(coupling, T=config.T, num_steps=config.num_steps, **factory_kwargs)
        self.readout = PhaseReadout()

    def forward(
        self, x: Tensor, num_steps: int | None = None, output_trajectory: bool = False
    ) -> KuramotoModelOutput:
        z = self.drive(x)
        if output_trajectory:
            traj = self.dynamics.forward_trajectory(z, num_steps=num_steps)
            theta = traj[-1]
        else:
            traj, theta = None, self.dynamics(z, num_steps=num_steps)
        features = self.readout(theta)
        return KuramotoModelOutput(drive=z, phases=theta, features=features, trajectory=traj)


class KuramotoForClassification(KuramotoPreTrainedModel):
    """(B, in_dim) -> (B, num_classes) logits. KuramotoModel plus a frozen random head.

    forward = forward_head(forward_features(x)), as in timm. forward returns the
    logits tensor so this is a drop-in replacement for any other classifier or
    policy network; for the intermediates, call the base model:
    `model.kuramoto(x)` returns a KuramotoModelOutput.
    """

    def __init__(
        self,
        config: KuramotoConfig,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(config)
        factory_kwargs = {"device": device, "dtype": dtype}
        self.num_classes = config.num_classes
        self.num_features = 2 * config.n
        self.kuramoto = KuramotoModel(config, generator=generator, **factory_kwargs)
        self.head = FrozenHead(self.num_features, config.num_classes, generator=generator, **factory_kwargs)

    def get_classifier(self) -> FrozenHead:
        return self.head

    def get_coupling(self) -> KuramotoCoupling:
        """The module holding K, the model's only trainable tensor."""
        return self.kuramoto.dynamics.coupling

    def forward_features(self, x: Tensor, num_steps: int | None = None) -> Tensor:
        """(B, in_dim) -> (B, 2n) readout features: everything before the head."""
        return self.kuramoto(x, num_steps=num_steps).features

    def forward_head(self, features: Tensor) -> Tensor:
        """(B, 2n) -> (B, num_classes)."""
        return self.head(features)

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_head(self.forward_features(x))


@torch.no_grad()
def calibrate(model: KuramotoForClassification, x_cal: Tensor, g: float = 1.0) -> None:
    """Set W's scale, g and tau in place from one calibration batch. Run once, at init.

    Args:
        model: freshly constructed classifier, modified in place.
        x_cal: (n_cal, in_dim) standardized calibration batch.
        g: drive gain in radians. Usable range is roughly [0.5, 1.5]; below ~0.3 the
            sin/cos features are effectively linear, above ~1.5 the tails wrap past
            pi and distinct inputs alias onto each other.

    This is deliberately one readable function rather than logic scattered through
    the layers' constructors, because it is what makes the model trainable at all.
    Each block owns its own data-dependent init (calibrate_); this function only
    fixes the order, which matters: each step measures the output of the last.
    """
    assert x_cal.ndim == 2 and x_cal.shape[1] == model.config.in_dim
    assert torch.isfinite(x_cal).all(), "calibration batch is not finite"

    # Step 2: normalize W empirically so the drive has unit variance by
    # construction. Everything downstream is then denominated in radians.
    model.kuramoto.drive.calibrate_(x_cal)

    # Step 3: because theta_0 = 0 the coupling term vanishes at t = 0, so
    # theta(t) ~= g * z * t and std_i(theta_i(T)) ~= g. g *is* the typical phase
    # excursion in radians at readout.
    model.kuramoto.dynamics.g.fill_(float(g))

    # Step 4: tau is the std of the raw pre-temperature logits, so logits have unit
    # scale at init. Dividing every logit by a positive constant cannot change the
    # argmax and therefore cannot change accuracy; it only conditions the
    # cross-entropy gradient. Measured with the W and g just set.
    model.head.calibrate_(model.forward_features(x_cal))


# Pre-pymoto KuramotoClassifier / KuramotoPolicy stored every tensor at the top
# level of one flat module.
_LEGACY_KEYS = {
    "K": "kuramoto.dynamics.coupling.K",
    "k_scale": "kuramoto.dynamics.coupling.k_scale",
    "g": "kuramoto.dynamics.g",
    "W": "kuramoto.drive.W",
    "H": "head.H",
    "tau": "head.tau",
}


def checkpoint_filter_fn(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    """Remap a pre-pymoto flat state dict (K, W, H, g, tau, k_scale) to this layout.

    timm's convention for loading weights saved by an older implementation.
    Already-converted state dicts pass through unchanged.
    """
    return {_LEGACY_KEYS.get(key, key): value for key, value in state_dict.items()}


def _create_kuramoto(
    *,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    **config_kwargs,
) -> KuramotoForClassification:
    return KuramotoForClassification(
        KuramotoConfig(**config_kwargs), generator=generator, device=device, dtype=dtype
    )


@register_model
def kuramoto_mnist(**kwargs) -> KuramotoForClassification:
    """Flattened MNIST: 784 -> 256 oscillators -> 10 classes, 10 Euler steps over T = 1."""
    model_args = dict(n=256, in_dim=784, num_classes=10, T=1.0, num_steps=10, k_scale=1.0)
    return _create_kuramoto(**dict(model_args, **kwargs))


@register_model
def kuramoto_cartpole(**kwargs) -> KuramotoForClassification:
    """CartPole policy: 4 observations -> 64 oscillators -> 2 action logits, 10 Euler steps."""
    model_args = dict(n=64, in_dim=4, num_classes=2, T=1.0, num_steps=10, k_scale=1.0)
    return _create_kuramoto(**dict(model_args, **kwargs))

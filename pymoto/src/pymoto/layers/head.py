"""The classifier head: a frozen random projection with a calibrated temperature."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class FrozenHead(nn.Module):
    """(B, in_features) -> (B, num_classes). logits = features @ H.T / tau, no bias.

    H and tau are buffers: they serialize with the checkpoint but never receive
    gradients. H is drawn on the CPU in float32, then cast to `device` / `dtype`.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.num_classes = num_classes

        # H: (num_classes, 2n) readout head. The final classifier layer, except it
        # is never trained -- a fixed random projection from the feature space to
        # the class scores, drawn before it has seen any data and no bias.
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
        self.register_buffer("H", torch.empty(num_classes, in_features, **factory_kwargs))

        # tau: logit temperature in logits = features @ H.T / tau. Set by
        # calibrate_() to the std of the raw pre-temperature logits, so logits have
        # unit scale at init.
        #
        # It cannot change accuracy: dividing every logit by the same positive
        # constant cannot change an argmax. It only conditions the cross-entropy
        # gradient, which is otherwise at the mercy of however H @ features happens
        # to scale -- raw logits with std ~30 saturate the softmax and the gradient
        # vanishes, std ~0.01 leaves the loss nearly flat at ln(10) = 2.303. Being
        # unable to affect predictions is what makes it safe to fit on data without
        # touching the attribution argument.
        self.register_buffer("tau", torch.tensor(1.0, **factory_kwargs))

        self.reset_parameters(generator)

    @torch.no_grad()
    def reset_parameters(self, generator: torch.Generator | None = None) -> None:
        self.H.copy_(torch.randn(self.num_classes, self.in_features, generator=generator))

    @torch.no_grad()
    def calibrate_(self, features: Tensor) -> None:
        """Set tau in place to the std of the raw (tau = 1) logits over the batch `features`."""
        self.tau.fill_(1.0)
        raw_std = self(features).std()
        assert raw_std > 0, "raw logits have zero variance; features may be constant"
        self.tau.fill_(float(raw_std))

    def forward(self, features: Tensor) -> Tensor:
        assert features.ndim == 2 and features.shape[1] == self.in_features
        return features @ self.H.T / self.tau

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, num_classes={self.num_classes}, tau={float(self.tau):.4g}"


class TrainableHead(FrozenHead):
    """FrozenHead with H as a Parameter instead of a buffer -- everything else identical.

    tau stays a fixed buffer, calibrated once at init as usual: it would be
    entirely redundant with a trainable H (both just rescale the same logits),
    so leaving it fixed removes an otherwise-unconstrained degree of freedom
    rather than adding a meaningful one.

    Exists only to test whether a task's ceiling comes from the frozen output
    stage rather than from K's capacity -- it's exactly pymoto.controls'
    linear_probe made permanent instead of a discarded measurement, so using
    it forfeits the "K is the only trainable tensor" attribution argument.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        nn.Module.__init__(self)
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.num_classes = num_classes
        self.H = nn.Parameter(torch.empty(num_classes, in_features, **factory_kwargs))
        self.register_buffer("tau", torch.tensor(1.0, **factory_kwargs))
        self.reset_parameters(generator)

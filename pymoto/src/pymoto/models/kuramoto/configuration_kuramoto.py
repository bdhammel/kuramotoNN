"""KuramotoConfig: everything needed to rebuild the Kuramoto network's architecture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from pymoto.configuration_utils import PretrainedConfig


@dataclass
class KuramotoConfig(PretrainedConfig):
    """Architecture of a Kuramoto network: drive -> dynamics -> readout -> head.

    Calibrated quantities -- W's scale, the drive gain g, the logit temperature
    tau -- are not here. They are fitted by calibrate() and stored in the state
    dict alongside K, so a saved model reproduces them exactly.

    Attributes:
        n: number of oscillators.
        in_dim: input dimension (784 for flattened MNIST, 4 for CartPole).
        num_classes: output logits.
        T: total integration time.
        num_steps: Euler steps over [0, T]; the depth of the weight-tied stack.
        k_scale: coupling strength, K_eff = k_scale * K. Fixed, never trained.
    """

    model_type: ClassVar[str] = "kuramoto"

    n: int = 256
    in_dim: int = 784
    num_classes: int = 10
    T: float = 1.0
    num_steps: int = 10
    k_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.n < 1 or self.in_dim < 1 or self.num_classes < 1:
            raise ValueError(f"n, in_dim and num_classes must be positive: {self}")
        if self.num_steps < 0:
            raise ValueError(f"num_steps must be >= 0, got {self.num_steps}")
        if self.T <= 0:
            raise ValueError(f"T must be positive, got {self.T}")

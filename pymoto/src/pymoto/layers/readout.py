"""The readout: phases -> rotation-invariant features. Parameter-free."""

from __future__ import annotations

import torch
from torch import Tensor, nn


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


class PhaseReadout(nn.Module):
    """(B, n) -> (B, 2n). Module form of readout_features; holds no state."""

    def forward(self, theta: Tensor) -> Tensor:
        features = readout_features(theta)
        assert features.shape == (theta.shape[0], 2 * theta.shape[1])
        return features

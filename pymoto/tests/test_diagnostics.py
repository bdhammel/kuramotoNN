import math

import pytest
import torch

from pymoto import KuramotoConfig, KuramotoForClassification, calibrate
from pymoto.diagnostics import (
    Diagnostics,
    assert_only_K_is_trainable,
    compute_diagnostics,
    format_diagnostics,
    jacobian_spectral_radius,
    participation_ratio,
    warn_on_diagnostics,
)


def test_participation_ratio_of_isotropic_noise_is_near_full_rank():
    features = torch.randn(4000, 20, generator=torch.Generator().manual_seed(0))
    assert participation_ratio(features) == pytest.approx(20, rel=0.05)


def test_participation_ratio_of_rank_one_features_is_one():
    features = torch.randn(100, 1) * torch.randn(1, 20)
    assert participation_ratio(features) == pytest.approx(1.0, rel=1e-6)


def test_jacobian_radius_ignores_the_diagonal():
    K = torch.randn(16, 16, dtype=torch.float64)
    K_junk = K + torch.diag(torch.randn(16, dtype=torch.float64))
    assert jacobian_spectral_radius(K) == pytest.approx(jacobian_spectral_radius(K_junk), abs=1e-12)


def test_compute_diagnostics_on_calibrated_model():
    config = KuramotoConfig(n=32, in_dim=12, num_classes=5, num_steps=10)
    model = KuramotoForClassification(config, generator=torch.Generator().manual_seed(0))
    x_cal = torch.randn(512, 12)
    calibrate(model, x_cal, g=1.0)

    diag = compute_diagnostics(model, x_cal)
    assert isinstance(diag, Diagnostics)
    assert all(math.isfinite(v) for v in diag.as_dict().values())
    assert 0.5 < diag.phase_std < 1.5
    assert diag == compute_diagnostics(model.kuramoto, x_cal)  # base or task model
    assert "rho" in format_diagnostics(diag)
    assert isinstance(warn_on_diagnostics(diag), list)


def test_attribution_guard_rejects_extra_trainables():
    model = KuramotoForClassification(KuramotoConfig(n=8, in_dim=3, num_classes=2))
    assert_only_K_is_trainable(model)
    model.head.H = torch.nn.Parameter(model.head.H.clone())
    with pytest.raises(AssertionError, match="only trainable"):
        assert_only_K_is_trainable(model)

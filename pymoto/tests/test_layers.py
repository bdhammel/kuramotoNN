import math

import pytest
import torch

from pymoto.layers import (
    FrozenDrive,
    FrozenHead,
    KuramotoCoupling,
    KuramotoDynamics,
    PhaseReadout,
    coupling,
    coupling_pairwise,
    euler_step,
    readout_features,
    rk4_step,
    rollout,
    trajectory,
)


def make_dynamics(n=16, T=1.0, num_steps=10, k_scale=1.0, seed=0) -> KuramotoDynamics:
    coupling = KuramotoCoupling(n, k_scale, generator=torch.Generator().manual_seed(seed))
    return KuramotoDynamics(coupling, T=T, num_steps=num_steps)


# ---------------------------------------------------------------- coupling


def test_coupling_matches_pairwise_reference():
    gen = torch.Generator().manual_seed(0)
    theta = torch.randn(16, 32, generator=gen) * 2
    K = torch.randn(32, 32, generator=gen) / math.sqrt(32)
    assert torch.allclose(coupling(theta, K), coupling_pairwise(theta, K), atol=1e-5)


def test_coupling_diagonal_is_a_null_direction():
    gen = torch.Generator().manual_seed(1)
    theta = torch.randn(8, 16, generator=gen)
    K = torch.randn(16, 16, generator=gen)
    K_junk = K + torch.diag(torch.randn(16, generator=gen) * 10)
    assert torch.allclose(coupling(theta, K), coupling(theta, K_junk), atol=1e-4)


def test_kuramoto_coupling_init():
    layer = KuramotoCoupling(64, k_scale=0.5, generator=torch.Generator().manual_seed(0))
    assert torch.equal(layer.K.diagonal(), torch.zeros(64))
    assert torch.equal(layer.K_eff(), 0.5 * layer.K)
    assert [name for name, _ in layer.named_parameters()] == ["K"]


def test_reset_parameters_redraws_from_generator():
    layer = KuramotoCoupling(8, generator=torch.Generator().manual_seed(3))
    first = layer.K.detach().clone()
    layer.reset_parameters(torch.Generator().manual_seed(3))
    assert torch.equal(layer.K, first)


# ---------------------------------------------------------------- factory kwargs


@pytest.mark.parametrize("layer_fn", [
    lambda **kw: KuramotoCoupling(8, 0.5, **kw),
    lambda **kw: FrozenDrive(5, 8, **kw),
    lambda **kw: FrozenHead(16, 3, **kw),
])
def test_dtype_and_seeded_init_are_independent(layer_fn):
    f32 = layer_fn(generator=torch.Generator().manual_seed(0))
    f64 = layer_fn(generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    for (name, a), (_, b) in zip(f32.state_dict().items(), f64.state_dict().items()):
        assert b.dtype == torch.float64, name
        assert torch.equal(a.double(), b), name  # same draw, only the storage dtype differs


def test_dynamics_follows_coupling_dtype():
    coupling = KuramotoCoupling(8, dtype=torch.float64)
    dyn = KuramotoDynamics(coupling, dtype=torch.float64)
    theta = dyn(torch.randn(3, 8, dtype=torch.float64))
    assert theta.dtype == torch.float64


# ---------------------------------------------------------------- integrators


def test_rollout_zero_steps_returns_initial_state():
    x0 = torch.randn(4, 3)
    assert rollout(euler_step, lambda x: x + 1, x0, 0.1, 0) is x0


def test_rk4_solves_linear_decay():
    x0 = torch.ones(1, 1, dtype=torch.float64)
    x = rollout(rk4_step, lambda x: -x, x0, 0.1, 10)
    assert abs(float(x) - math.exp(-1.0)) < 1e-6


def test_euler_is_first_order():
    f = torch.sin
    x0 = torch.full((1, 1), 0.5, dtype=torch.float64)
    exact = rollout(rk4_step, f, x0, 1e-3, 1000)
    coarse = (rollout(euler_step, f, x0, 0.1, 10) - exact).abs()
    fine = (rollout(euler_step, f, x0, 0.01, 100) - exact).abs()
    assert fine < coarse / 5  # 10x smaller step, ~10x smaller error


def test_trajectory_ends_where_rollout_ends():
    f = torch.cos
    x0 = torch.randn(4, 3)
    traj = trajectory(euler_step, f, x0, 0.1, 7)
    assert traj.shape == (8, 4, 3)
    assert torch.equal(traj[0], x0)
    assert torch.equal(traj[-1], rollout(euler_step, f, x0, 0.1, 7))


# ---------------------------------------------------------------- dynamics


def test_dynamics_zero_steps_leaves_phases_at_zero():
    dyn = make_dynamics()
    z = torch.randn(4, 16)
    assert torch.equal(dyn(z, num_steps=0), torch.zeros(4, 16))
    assert dyn.h == 0.1


def test_dynamics_without_coupling_rotates_uniformly():
    dyn = make_dynamics(T=2.0, num_steps=7, k_scale=0.0)
    z = torch.randn(4, 16)
    assert torch.allclose(dyn(z), dyn.g * z * 2.0, atol=1e-5)


def test_dynamics_trajectory():
    dyn = make_dynamics(num_steps=5)
    z = torch.randn(4, 16)
    traj = dyn.forward_trajectory(z)
    assert traj.shape == (6, 4, 16)
    assert torch.equal(traj[0], torch.zeros(4, 16))
    assert torch.equal(traj[-1], dyn(z))


def _grad_K_and_saved_bytes(dyn: KuramotoDynamics, z: torch.Tensor) -> tuple[torch.Tensor, int]:
    saved = []

    def pack(t):
        saved.append(t.numel() * t.element_size())
        return t

    dyn.coupling.K.grad = None
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        loss = dyn(z).sin().sum()
    loss.backward()
    return dyn.coupling.K.grad.clone(), sum(saved)


def test_grad_checkpointing_same_gradient_less_memory():
    dyn = make_dynamics(n=32, num_steps=20)
    z = torch.randn(64, 32)

    grad_plain, bytes_plain = _grad_K_and_saved_bytes(dyn, z)
    dyn.grad_checkpointing = True
    grad_ckpt, bytes_ckpt = _grad_K_and_saved_bytes(dyn, z)

    assert torch.equal(grad_plain, grad_ckpt)
    assert bytes_ckpt < bytes_plain / 3


def test_grad_checkpointing_is_inert_without_grad():
    dyn = make_dynamics()
    dyn.grad_checkpointing = True
    z = torch.randn(4, 16)
    with torch.no_grad():
        assert torch.equal(dyn(z), make_dynamics()(z))


# ---------------------------------------------------------------- readout / frozen layers


def test_readout_is_invariant_to_global_rotation():
    theta = torch.randn(8, 16)
    assert torch.allclose(readout_features(theta), readout_features(theta + 1.234), atol=1e-6)
    assert PhaseReadout()(theta).shape == (8, 32)


def test_frozen_layers_have_no_parameters():
    for layer in [FrozenDrive(10, 4), FrozenHead(8, 3), PhaseReadout()]:
        assert list(layer.parameters()) == []
    assert FrozenDrive(10, 4)(torch.randn(2, 10)).shape == (2, 4)
    assert FrozenHead(8, 3)(torch.randn(2, 8)).shape == (2, 3)


def test_drive_calibrate_gives_unit_std():
    drive = FrozenDrive(20, 8)
    x = torch.randn(256, 20) * 7
    drive.calibrate_(x)
    assert torch.allclose(drive(x).std(), torch.tensor(1.0), atol=1e-5)


def test_head_calibrate_gives_unit_std_logits_and_keeps_argmax():
    head = FrozenHead(16, 5)
    features = torch.randn(256, 16)
    before = head(features).argmax(-1)
    head.calibrate_(features)
    assert torch.allclose(head(features).std(), torch.tensor(1.0), atol=1e-5)
    assert torch.equal(head(features).argmax(-1), before)

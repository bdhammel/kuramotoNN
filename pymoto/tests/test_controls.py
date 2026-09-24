import torch
from torch.utils.data import DataLoader, TensorDataset

from pymoto import KuramotoConfig, KuramotoForClassification, calibrate
from pymoto.controls import collect_features, linear_probe, with_coupling, with_num_steps, with_solver
from pymoto.layers import euler_step, rk4_step


def small_model() -> KuramotoForClassification:
    config = KuramotoConfig(n=32, in_dim=12, num_classes=3, num_steps=6)
    model = KuramotoForClassification(config, generator=torch.Generator().manual_seed(0))
    calibrate(model, torch.randn(256, 12))
    return model


def test_with_num_steps_severs_the_input_and_leaves_the_original_alone():
    model = small_model()
    zero = with_num_steps(model, 0)
    features = zero.forward_features(torch.randn(10, 12) * 5)
    assert torch.equal(features, features[:1].expand_as(features))
    assert zero.config.num_steps == zero.kuramoto.config.num_steps == 0
    assert model.config.num_steps == model.kuramoto.dynamics.num_steps == 6


def test_with_coupling_swaps_only_K():
    model = small_model()
    K_new = torch.randn(32, 32)
    variant = with_coupling(model, K_new)
    assert torch.equal(variant.get_coupling().K, K_new)
    assert not torch.equal(model.get_coupling().K, K_new)
    for key, value in model.state_dict().items():
        if not key.endswith("coupling.K"):
            assert torch.equal(value, variant.state_dict()[key]), key


def test_with_solver_shares_weights_and_reduces_to_the_model():
    model = small_model()
    x = torch.randn(8, 12)
    same = with_solver(model, euler_step, refine=1)
    assert torch.equal(same(x), model(x))
    assert same.model is model

    fine = with_solver(model, rk4_step, refine=10)
    assert fine.num_steps == 60
    assert fine(x).shape == (8, 3)


def test_linear_probe_reads_features():
    model = small_model()
    x = torch.randn(300, 12)
    y = (x[:, 0] > 0).long()
    loader = DataLoader(TensorDataset(x, y), batch_size=100)
    features, labels = collect_features(model, loader, torch.device("cpu"))
    assert features.shape == (300, 64) and torch.equal(labels, y)
    result = linear_probe(model, loader, loader, torch.device("cpu"), epochs=5)
    assert set(result) == {"train_acc", "test_acc"}
    assert 0.0 <= result["test_acc"] <= 1.0

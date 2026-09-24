import pytest
import torch

import pymoto
from pymoto import (
    KuramotoConfig,
    KuramotoForClassification,
    KuramotoModel,
    KuramotoModelOutput,
    calibrate,
    checkpoint_filter_fn,
    create_model,
    list_models,
)
from pymoto.diagnostics import assert_only_K_is_trainable


def small_model(**kwargs) -> KuramotoForClassification:
    args = dict(n=32, in_dim=12, num_classes=5, num_steps=4) | kwargs
    return KuramotoForClassification(KuramotoConfig(**args), generator=torch.Generator().manual_seed(0))


def test_registry():
    assert list_models() == ["kuramoto_cartpole", "kuramoto_mnist"]
    assert list_models("*mnist") == ["kuramoto_mnist"]
    model = create_model("kuramoto_cartpole", num_steps=3)
    assert (model.config.in_dim, model.config.num_classes, model.config.n) == (4, 2, 64)
    assert model.config.num_steps == 3
    with pytest.raises(ValueError, match="unknown model"):
        create_model("resnet50")


def test_only_K_is_trainable():
    model = small_model()
    assert_only_K_is_trainable(model)
    assert [name for name, _ in model.named_parameters()] == ["kuramoto.dynamics.coupling.K"]
    assert model.get_coupling().K is model.kuramoto.dynamics.coupling.K


def test_state_dict_follows_module_tree():
    assert sorted(small_model().state_dict()) == [
        "head.H",
        "head.tau",
        "kuramoto.drive.W",
        "kuramoto.dynamics.coupling.K",
        "kuramoto.dynamics.coupling.k_scale",
        "kuramoto.dynamics.g",
    ]


def test_forward_is_head_of_features():
    model = small_model()
    x = torch.randn(6, 12)
    features = model.forward_features(x)
    assert features.shape == (6, model.num_features) == (6, 64)
    assert torch.equal(model(x), model.forward_head(features))
    out = model.kuramoto(x)
    assert isinstance(out, KuramotoModelOutput)
    assert out.drive.shape == out.phases.shape == (6, 32)
    assert torch.equal(out.features, features)


def test_output_trajectory():
    model = small_model()
    x = torch.randn(6, 12)
    out = model.kuramoto(x, output_trajectory=True)
    assert out.trajectory.shape == (5, 6, 32)  # num_steps + 1
    assert torch.equal(out.trajectory[-1], out.phases)
    plain = model.kuramoto(x)
    assert plain.trajectory is None
    assert torch.equal(plain.features, out.features)


def test_set_grad_checkpointing_reaches_the_dynamics():
    model = small_model()
    model.set_grad_checkpointing()
    assert model.kuramoto.dynamics.grad_checkpointing
    model.set_grad_checkpointing(False)
    assert not model.kuramoto.dynamics.grad_checkpointing


def test_float64_model_has_the_float32_init():
    a = create_model("kuramoto_cartpole", generator=torch.Generator().manual_seed(0))
    b = create_model("kuramoto_cartpole", generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    for key, value in b.state_dict().items():
        assert value.dtype == torch.float64, key
        assert torch.equal(value, a.state_dict()[key].double()), key
    assert b(torch.randn(3, 4, dtype=torch.float64)).dtype == torch.float64


def test_base_model_prefix():
    model = small_model()
    assert model.base_model is model.kuramoto
    assert model.kuramoto.base_model is model.kuramoto


def test_zero_steps_severs_the_input():
    model = small_model()
    features = model.forward_features(torch.randn(10, 12) * 5, num_steps=0)
    assert torch.equal(features, features[:1].expand_as(features))


def test_calibrate_sets_unit_scales():
    model = small_model()
    x_cal = torch.randn(512, 12)
    calibrate(model, x_cal, g=0.8)
    assert torch.allclose(model.kuramoto.drive(x_cal).std(), torch.tensor(1.0), atol=1e-5)
    assert float(model.kuramoto.dynamics.g) == pytest.approx(0.8)
    assert torch.allclose(model(x_cal).std(), torch.tensor(1.0), atol=1e-5)


def test_seeded_construction_is_deterministic():
    a, b = small_model(), small_model()
    for key, value in a.state_dict().items():
        assert torch.equal(value, b.state_dict()[key]), key


def test_save_and_load_pretrained_roundtrip(tmp_path):
    model = small_model()
    calibrate(model, torch.randn(256, 12))
    model.save_pretrained(tmp_path / "ckpt")
    assert {p.name for p in (tmp_path / "ckpt").iterdir()} == {"config.json", "pytorch_model.bin"}

    loaded = KuramotoForClassification.from_pretrained(tmp_path / "ckpt")
    assert loaded.config == model.config
    x = torch.randn(8, 12)
    assert torch.equal(loaded(x), model(x))


def test_config_roundtrip_is_strict():
    config = KuramotoConfig(n=8, in_dim=3, num_classes=2)
    assert KuramotoConfig.from_dict(config.to_dict()) == config
    with pytest.raises(ValueError, match="unknown"):
        KuramotoConfig.from_dict({**config.to_dict(), "lr": 1e-3})
    with pytest.raises(ValueError, match="num_steps"):
        KuramotoConfig(num_steps=-1)


def test_model_rejects_wrong_config_type():
    with pytest.raises(TypeError):
        KuramotoModel(object())


def test_checkpoint_filter_fn_loads_legacy_flat_state_dict():
    model = small_model()
    legacy = {
        "K": torch.randn(32, 32),
        "W": torch.randn(32, 12),
        "H": torch.randn(5, 64),
        "g": torch.tensor(1.1),
        "tau": torch.tensor(7.0),
        "k_scale": torch.tensor(0.9),
    }
    model.load_state_dict(checkpoint_filter_fn(legacy))  # strict
    assert torch.equal(model.get_coupling().K, legacy["K"])
    assert torch.equal(model.head.tau, legacy["tau"])
    # Already-converted dicts pass through.
    assert checkpoint_filter_fn(model.state_dict()).keys() == model.state_dict().keys()


def test_public_api():
    assert pymoto.__version__ == "0.1.0"
    for name in pymoto.__all__:
        assert hasattr(pymoto, name), name

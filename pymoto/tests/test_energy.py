import argparse
import dataclasses

import pytest
import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

from pymoto import KuramotoConfig, KuramotoForClassification, calibrate
from pymoto.controls import with_coupling, with_num_steps
from pymoto.energy import (
    GOOGLE_7NM_INT8,
    OscillatorHardware,
    Workload,
    add_energy_args,
    count_couplers,
    digital_energy,
    digital_workload,
    energy_report,
    format_energy_report,
    kuramoto_workload,
    mlp_workload,
    override,
    physical_energy,
    report_from_args,
    total_workload,
)


def small_model() -> KuramotoForClassification:
    config = KuramotoConfig(n=32, in_dim=12, num_classes=3, num_steps=6)
    model = KuramotoForClassification(config, generator=torch.Generator().manual_seed(0))
    calibrate(model, torch.randn(256, 12))
    return model


def test_kuramoto_macs_match_the_forward_pass():
    # FlopCounterMode counts the matmuls pymoto actually runs, at 2 flops per MAC.
    model = small_model()
    with FlopCounterMode(display=False) as counter:
        model(torch.randn(1, 12))
    assert total_workload(kuramoto_workload(model.config)).macs == counter.get_total_flops() // 2


def test_linear_workload_matches_mlp_workload_and_the_forward_pass():
    mlp = nn.Sequential(nn.Linear(4, 128), nn.ReLU(), nn.Linear(128, 2))
    stages = digital_workload(mlp)
    assert list(stages.values()) == list(mlp_workload((4, 128, 2)).values())
    assert total_workload(stages).weights == sum(p.numel() for p in mlp.parameters())
    with FlopCounterMode(display=False) as counter:
        mlp(torch.randn(1, 4))
    assert total_workload(stages).macs == counter.get_total_flops() // 2


def test_linear_workload_refuses_layers_it_cannot_cost():
    with pytest.raises(TypeError):
        digital_workload(nn.Sequential(nn.Conv1d(1, 4, 3)))


def test_K_is_fetched_once_and_reread_every_step():
    coupling = kuramoto_workload(KuramotoConfig(n=16, in_dim=4, num_classes=2, num_steps=5))["coupling"]
    assert coupling == Workload(macs=5 * 2 * 256, weights=256, weight_reads=5 * 256)


def test_batching_amortizes_memory_but_not_compute():
    w = total_workload(kuramoto_workload(KuramotoConfig()))
    one = digital_energy(w, GOOGLE_7NM_INT8, "dram", batch_size=1)
    many = digital_energy(w, GOOGLE_7NM_INT8, "dram", batch_size=8)
    assert many["compute"] == one["compute"]
    assert many["memory"] == pytest.approx(one["memory"] / 8)
    assert one["total"] == pytest.approx(one["compute"] + one["memory"])


def test_physical_core_depends_on_T_not_num_steps():
    model = small_model()
    core = physical_energy(model)["core"]
    assert physical_energy(with_num_steps(model, 60))["core"] == pytest.approx(core)
    longer = KuramotoForClassification(dataclasses.replace(model.config, T=2.0))
    assert physical_energy(longer)["core"] > core


def test_couplers_skip_the_diagonal_and_small_entries():
    model = small_model()
    assert count_couplers(model) == 32 * 31
    K = model.get_coupling().K.detach().clone()
    K[:, 16:] = 0.0
    sparse = with_coupling(model, K)
    assert count_couplers(sparse) == 32 * 16 - 16
    dense, thin = physical_energy(model), physical_energy(sparse)
    assert thin["couplers"] < dense["couplers"] and thin["wires"] < dense["wires"]
    assert thin["oscillators"] == dense["oscillators"]


def test_weight_source_only_changes_the_digital_stages():
    model = small_model()
    sram, dram, seed = (physical_energy(model, weight_source=s) for s in ("sram", "dram", "seed"))
    assert sram["core"] == dram["core"] == seed["core"]
    assert seed["drive"] < sram["drive"] < dram["drive"]


def test_report_for_kuramoto_and_mlp():
    model = small_model()
    report = energy_report(model, baseline=mlp_workload((12, 32, 3)))
    assert report["energy/sim_over_core"] > 1.0
    assert 0.0 < report["energy/drive_head_share"] < 1.0
    assert "energy/baseline_over_physical_seed" in report
    assert all(isinstance(v, float) for v in report.values())
    assert "nJ" in format_energy_report(report)

    mlp_report = energy_report(nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2)))
    assert not any("physical" in key for key in mlp_report)
    assert format_energy_report(mlp_report).splitlines()[2].startswith("digital ")


def test_report_from_args_reads_the_shared_flags():
    parser = argparse.ArgumentParser()
    add_energy_args(parser)
    model = small_model()

    report, table = report_from_args(model, parser.parse_args([]), default_baseline=(12, 32, 3))
    assert "MLP 12-32-3" in table and "energy/baseline_macs" in report

    args = parser.parse_args(["--energy-baseline", "none", "--hw", "cycles_per_unit_time=10",
                              "--energy-costs", "45nm-fp32"])
    report, table = report_from_args(model, args, default_baseline=(12, 32, 3))
    assert "energy/baseline_macs" not in report and report["energy/cycles"] == 10.0
    assert "45nm fp32" in table


def test_override_parses_fields_and_rejects_unknown_ones():
    hw = override(OscillatorHardware(), ["cycles_per_unit_time=250", "conversions_per_oscillator=1"])
    assert hw.cycles_per_unit_time == 250.0 and hw.conversions_per_oscillator == 1
    with pytest.raises(ValueError):
        override(hw, ["not_a_field=1"])

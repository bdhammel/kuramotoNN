"""Energy per inference: a model simulated on a digital processor vs run as physical oscillators.

Every number here is (count of events) x (energy per event), summed. The two
accountings differ in what counts as an event:

    digital     multiply-accumulates, plus weight reads from on-chip SRAM or off-chip
                DRAM/HBM. Per sample, batch-size amortized. This is how the MLP
                baselines are costed, and how the Kuramoto network is costed when it
                is simulated with Euler steps.
    physical    oscillator-cycles, coupler-seconds, coupling-wire toggles and the
                data converters at either end. K lives in the coupling fabric and is
                never read from memory; num_steps does not exist, the physics
                integrates the flow for T model-time units. The frozen drive W and
                head H are still digital matmuls and are costed as such.

    DigitalCosts, OscillatorHardware    per-event energies (presets / assumptions)
    Workload                            per-stage event counts for one sample
    kuramoto_workload, linear_workload, mlp_workload, digital_workload
    digital_energy(workload, ...)       -> {"compute", "memory", "total"} in joules
    physical_energy(model, ...)         -> per-term breakdown in joules
    energy_report(model, ...)           -> flat dict[str, float] for JSON / wandb, in nJ
    format_energy_report(report)        -> stdout table
    add_energy_args, report_from_args   the shared command-line flags, for eval scripts

    python -m pymoto.energy kuramoto_mnist --energy-baseline 784,256,10
    python -m pymoto.energy kuramoto_mnist --set n=1024 --hw cycles_per_unit_time=200

The digital presets are published circuit figures. OscillatorHardware's defaults are
not: they are order-of-magnitude placeholders for CMOS ring oscillators, and the
report says so. Replace them with numbers for the hardware you have in mind.
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass
from typing import Iterable, Literal, Mapping

import torch
from torch import nn

from pymoto.models.kuramoto import KuramotoConfig, KuramotoForClassification

Memory = Literal["sram", "dram"]
WeightSource = Literal["sram", "dram", "seed"]


@dataclass(frozen=True)
class DigitalCosts:
    """Energy per event on a digital processor, in joules. A MAC costs mul_j + add_j.

    Memory is per byte, so bytes_per_weight carries the precision. Activations are
    assumed to stay in registers or local buffers and are not counted: per sample they
    are O(n) against the O(n^2) weights.
    """

    name: str
    mul_j: float
    add_j: float
    bytes_per_weight: int
    sram_j_per_byte: float  # on-chip, ~1 MB arrays
    dram_j_per_byte: float  # off-chip DRAM / HBM


# Horowitz, "Computing's energy problem", ISSCC 2014, 45 nm at 0.9 V. Memory: 100 pJ
# per 64-bit read from a 1 MB SRAM, 1.3-2.6 nJ per 64-bit DRAM read (low end used).
HOROWITZ_45NM_FP32 = DigitalCosts(
    "45nm fp32 (Horowitz 2014)", mul_j=3.7e-12, add_j=0.9e-12, bytes_per_weight=4,
    sram_j_per_byte=100e-12 / 8, dram_j_per_byte=1.3e-9 / 8,
)
# Same table: 8-bit multiply, accumulated in 32 bits.
HOROWITZ_45NM_INT8 = dataclasses.replace(
    HOROWITZ_45NM_FP32, name="45nm int8 (Horowitz 2014)", mul_j=0.2e-12, add_j=0.1e-12, bytes_per_weight=1,
)
# Google, ISCA 2021 (doi:10.1109/ISCA52012.2021.00010), 7 nm, as quoted by unconv.ai's
# "How to improve AI energy efficiency by 1000x": 0.07 pJ per 8-bit multiply, 2 pJ per byte from a 1 MB
# SRAM, 39 pJ per byte from HBM. The source counts multiplies only, so adds are 0 here.
GOOGLE_7NM_INT8 = DigitalCosts(
    "7nm int8 (Google ISCA 2021)", mul_j=0.07e-12, add_j=0.0, bytes_per_weight=1,
    sram_j_per_byte=2e-12, dram_j_per_byte=39e-12,
)

DIGITAL_PRESETS: dict[str, DigitalCosts] = {
    "7nm-int8": GOOGLE_7NM_INT8,
    "45nm-int8": HOROWITZ_45NM_INT8,
    "45nm-fp32": HOROWITZ_45NM_FP32,
}


@dataclass(frozen=True)
class OscillatorHardware:
    """Energy per event for a physical oscillator network. Defaults are placeholders.

    The one conversion that matters is time. The model runs for T units of model
    time; one unit is `cycles_per_unit_time` carrier cycles. Real oscillators obey
    Kuramoto phase dynamics only under weak coupling -- coupling rate well below the
    oscillator's damping rate omega_0 / 2Q -- so a unit of model time is tens to
    hundreds of carrier cycles, not one. The phase-noise and coupling-strength limits
    of a real design set this number, and every time-proportional term scales with it.

    Defaults are order-of-magnitude CMOS ring-oscillator figures, not measurements.
    """

    carrier_hz: float = 1e9
    cycles_per_unit_time: float = 100.0
    energy_per_cycle_j: float = 50e-15  # per oscillator per carrier cycle; ring osc ~10-100 fJ
    # Per active coupler. Asymmetric K needs active (buffered, non-reciprocal)
    # couplers, each drawing bias current; see diagnostics' k_asymmetry.
    coupler_power_w: float = 1e-6
    # Each coupler adds one pitch of line that its source oscillator charges and
    # discharges once per cycle (rail-to-rail CMOS): all-to-all coupling moves the
    # oscillator states across the chip continuously, which is data movement in
    # another form.
    wire_cap_f_per_um: float = 0.2e-15
    coupler_pitch_um: float = 10.0
    supply_v: float = 0.8
    dac_j_per_conversion: float = 1e-12  # sets each oscillator's drive g*z_i once
    adc_j_per_conversion: float = 1e-12  # ~8-bit converter
    # [sin, cos] of the phase is I/Q demodulation against a reference: two conversions.
    conversions_per_oscillator: int = 2
    # Regenerating a frozen random weight from its seed instead of reading it. Not
    # covered by the sources above; an LFSR emitting +-1 entries costs well under a MAC.
    rng_j_per_weight: float = 0.0


@dataclass(frozen=True)
class Workload:
    """Digital event counts for one sample through one stage.

    macs          multiply-accumulates
    weights       distinct weight scalars touched: the off-chip fetch, once
    weight_reads  on-chip weight reads counting reuse. A batch-1 matrix-vector product
                  reads each weight once per use; K is read once per Euler step and
                  serves both of that step's coupling matmuls, so reads = MACs / 2.
    """

    macs: int
    weights: int
    weight_reads: int

    def __add__(self, other: Workload) -> Workload:
        return Workload(
            self.macs + other.macs, self.weights + other.weights, self.weight_reads + other.weight_reads
        )


def total_workload(workload: Workload | Mapping[str, Workload]) -> Workload:
    """A per-stage workload summed into one; a single Workload passes through."""
    if isinstance(workload, Workload):
        return workload
    return sum(workload.values(), Workload(0, 0, 0))


def _dense(in_features: int, out_features: int, bias: bool = False) -> Workload:
    weights = in_features * out_features + (out_features if bias else 0)
    return Workload(in_features * out_features, weights, weights)


def kuramoto_workload(config: KuramotoConfig) -> dict[str, Workload]:
    """Per-stage counts for the network as pymoto computes it, Euler steps and all.

    The coupling is two (1, n) @ (n, n) products per step (layers.coupling), K's
    diagonal included since the matmul is dense. Elementwise work and the 2n sin/cos
    per step are O(n) and left out.
    """
    n, steps = config.n, config.num_steps
    return {
        "drive": _dense(config.in_dim, n),
        "coupling": Workload(steps * 2 * n * n, n * n, steps * n * n),
        "head": _dense(2 * n, config.num_classes),
    }


def linear_workload(model: nn.Module) -> dict[str, Workload]:
    """Per-layer counts for a stack of nn.Linear, each applied once per forward (an MLP)."""
    stages: dict[str, Workload] = {}
    for name, module in model.named_modules():
        if not any(True for _ in module.parameters(recurse=False)):
            continue
        if not isinstance(module, nn.Linear):
            raise TypeError(f"{name}: {type(module).__name__} is not costed; only nn.Linear stacks are")
        stages[name] = _dense(module.in_features, module.out_features, module.bias is not None)
    if not stages:
        raise TypeError(f"{type(model).__name__} has no nn.Linear layers to cost")
    return stages


def mlp_workload(dims: Iterable[int], bias: bool = True) -> dict[str, Workload]:
    """Counts for a reference MLP dims[0] -> ... -> dims[-1] that need not exist, e.g. (784, 256, 10)."""
    dims = list(dims)
    return {f"layer{i}": _dense(a, b, bias) for i, (a, b) in enumerate(zip(dims, dims[1:]))}


def digital_workload(model: nn.Module) -> dict[str, Workload]:
    """Per-stage counts for a Kuramoto network or an MLP."""
    if isinstance(model, KuramotoForClassification):
        return kuramoto_workload(model.config)
    return linear_workload(model)


def digital_energy(
    workload: Workload | Mapping[str, Workload],
    costs: DigitalCosts = GOOGLE_7NM_INT8,
    memory: Memory = "sram",
    batch_size: int = 1,
) -> dict[str, float]:
    """Joules per sample: {"compute", "memory", "total"}.

    memory="sram"  the whole model is resident on-chip; every weight read is an SRAM read.
    memory="dram"  weights are fetched once from DRAM/HBM, and any reuse beyond that
                   first read (K on later Euler steps) comes from SRAM.

    Weight traffic is shared by the batch, so it divides by batch_size; compute does not.
    """
    w = total_workload(workload)
    compute = w.macs * (costs.mul_j + costs.add_j)
    if memory == "sram":
        traffic = w.weight_reads * costs.sram_j_per_byte
    elif memory == "dram":
        traffic = w.weights * costs.dram_j_per_byte + (w.weight_reads - w.weights) * costs.sram_j_per_byte
    else:
        raise ValueError(f"memory must be 'sram' or 'dram', got {memory!r}")
    mem = traffic * costs.bytes_per_weight / batch_size
    return {"compute": compute, "memory": mem, "total": compute + mem}


def _stage_energy(
    workload: Workload, source: WeightSource, costs: DigitalCosts, hw: OscillatorHardware, batch_size: int
) -> float:
    """A digital stage of the physical network: its weights may also be regenerated from a seed."""
    if source == "seed":
        return workload.macs * (costs.mul_j + costs.add_j) + workload.weights * hw.rng_j_per_weight
    return digital_energy(workload, costs, source, batch_size)["total"]


@torch.no_grad()
def count_couplers(model: KuramotoForClassification, threshold: float = 0.0) -> int:
    """Physical couplers needed for K: off-diagonal entries with |K_eff_ij| > threshold.

    The diagonal is a null direction of the dynamics (layers.coupling.effective_coupling)
    and would not be built. A sparse K needs fewer couplers and shorter wires.
    """
    K_eff = model.get_coupling().K_eff()
    off_diagonal = ~torch.eye(K_eff.shape[0], dtype=torch.bool, device=K_eff.device)
    return int(((K_eff.abs() > threshold) & off_diagonal).sum())


def physical_energy(
    model: KuramotoForClassification,
    hw: OscillatorHardware = OscillatorHardware(),
    costs: DigitalCosts = GOOGLE_7NM_INT8,
    weight_source: WeightSource = "sram",
    batch_size: int = 1,
    coupler_threshold: float = 0.0,
) -> dict[str, float]:
    """Joules per sample for the network run as physical oscillators.

    Terms, in the order a sample meets them:
        drive            x @ W.T, digital, W from `weight_source` ("seed": regenerated)
        set_frequencies  one DAC write per oscillator to set its drive g*z_i
        oscillators      n oscillators x cycles
        couplers         active couplers x run time
        wires            coupler lines toggled every cycle
        readout          conversions_per_oscillator ADC conversions per oscillator
        head             features @ H.T, digital, H from `weight_source`
    plus "core" (everything physical: set_frequencies through readout) and "total".

    Not counted: writing K into the coupling fabric (once per model, amortized over
    every inference) and idle leakage between inferences.
    """
    config = model.config
    stages = kuramoto_workload(config)
    cycles = config.T * hw.cycles_per_unit_time
    run_time = cycles / hw.carrier_hz
    n_couplers = count_couplers(model, coupler_threshold)

    terms = {
        "drive": _stage_energy(stages["drive"], weight_source, costs, hw, batch_size),
        "set_frequencies": config.n * hw.dac_j_per_conversion,
        "oscillators": config.n * hw.energy_per_cycle_j * cycles,
        "couplers": n_couplers * hw.coupler_power_w * run_time,
        "wires": n_couplers * hw.coupler_pitch_um * hw.wire_cap_f_per_um * hw.supply_v**2 * cycles,
        "readout": config.n * hw.conversions_per_oscillator * hw.adc_j_per_conversion,
        "head": _stage_energy(stages["head"], weight_source, costs, hw, batch_size),
    }
    terms["core"] = sum(v for k, v in terms.items() if k not in ("drive", "head"))
    terms["total"] = terms["core"] + terms["drive"] + terms["head"]
    return terms


_NJ = 1e9


def energy_report(
    model: nn.Module,
    costs: DigitalCosts = GOOGLE_7NM_INT8,
    hw: OscillatorHardware = OscillatorHardware(),
    baseline: Mapping[str, Workload] | None = None,
    batch_size: int = 1,
    coupler_threshold: float = 0.0,
) -> dict[str, float]:
    """Flat dict of energy estimates for `model`, energies in nJ per sample.

    Always: the digital cost of `model` with weights in SRAM and from DRAM. For a
    Kuramoto network, also the physical breakdown and three totals that differ only in
    where the digital W and H come from (SRAM, DRAM, regenerated from their seed),
    plus the two ratios that decide whether the physics is worth building:

        sim_over_core      digital simulation (SRAM) / physical core. Should be >> 1:
                           a circuit cheaper to simulate than to run is not worth building.
        drive_head_share   fraction of the physical total (SRAM) spent on the digital W
                           and H. Near 1 means the oscillators cannot help until the
                           drive stops reading W from memory (Amdahl).

    `baseline` (e.g. mlp_workload((784, 256, 10))) adds its digital cost and
    physical-vs-baseline ratios. It should be a model of the same accuracy for the
    ratios to mean anything; that is the caller's to establish.
    """
    stages = digital_workload(model)
    w = total_workload(stages)
    r: dict[str, float] = {
        "energy/batch_size": float(batch_size),
        "energy/macs": float(w.macs),
        "energy/weights": float(w.weights),
    }
    for memory in ("sram", "dram"):
        r[f"energy/digital_{memory}_nj"] = digital_energy(w, costs, memory, batch_size)["total"] * _NJ

    if isinstance(model, KuramotoForClassification):
        config = model.config
        cycles = config.T * hw.cycles_per_unit_time
        r["energy/n_couplers"] = float(count_couplers(model, coupler_threshold))
        r["energy/cycles"] = cycles
        r["energy/run_time_ns"] = cycles / hw.carrier_hz * 1e9
        by_source = {
            source: physical_energy(model, hw, costs, source, batch_size, coupler_threshold)
            for source in ("sram", "dram", "seed")
        }
        for term, value in by_source["sram"].items():
            if term not in ("drive", "head", "total"):
                r[f"energy/physical_{term}_nj"] = value * _NJ
        for source, terms in by_source.items():
            r[f"energy/physical_total_{source}_nj"] = terms["total"] * _NJ
        sram = by_source["sram"]
        r["energy/sim_over_core"] = r["energy/digital_sram_nj"] / r["energy/physical_core_nj"]
        r["energy/drive_head_share"] = (sram["drive"] + sram["head"]) / sram["total"]

    if baseline is not None:
        b = total_workload(baseline)
        r["energy/baseline_macs"] = float(b.macs)
        for memory in ("sram", "dram"):
            r[f"energy/baseline_{memory}_nj"] = digital_energy(b, costs, memory, batch_size)["total"] * _NJ
        if "energy/physical_total_sram_nj" in r:
            for source in ("sram", "seed"):
                r[f"energy/baseline_over_physical_{source}"] = (
                    r["energy/baseline_sram_nj"] / r[f"energy/physical_total_{source}_nj"]
                )
    return r


def format_energy_report(
    report: dict[str, float],
    costs: DigitalCosts = GOOGLE_7NM_INT8,
    baseline_name: str = "baseline",
) -> str:
    """Render an energy_report for stdout. Pass the same costs / baseline_name used to build it."""
    rows: list[tuple[str, str, str]] = []

    def nj(label: str, key: str, note: str = "") -> None:
        if key in report:
            rows.append((label, f"{report[key]:12.4g} nJ", note))

    physical = "energy/physical_core_nj" in report
    rows.append((
        "digital simulation" if physical else "digital", "",
        f"{report['energy/macs']:.3g} MACs, {report['energy/weights']:.3g} weights",
    ))
    nj("  weights in SRAM", "energy/digital_sram_nj")
    nj("  weights from DRAM/HBM", "energy/digital_dram_nj")

    if physical:
        rows.append((
            "physical oscillators", "",
            f"{report['energy/n_couplers']:.0f} couplers, {report['energy/cycles']:.4g} cycles "
            f"= {report['energy/run_time_ns']:.4g} ns",
        ))
        nj("  set frequencies (DAC)", "energy/physical_set_frequencies_nj")
        nj("  oscillators", "energy/physical_oscillators_nj")
        nj("  couplers", "energy/physical_couplers_nj")
        nj("  wires", "energy/physical_wires_nj")
        nj("  readout (ADC)", "energy/physical_readout_nj")
        nj("  core", "energy/physical_core_nj",
           f"simulation costs {report['energy/sim_over_core']:.3g}x the core")
        nj("  + W, H in SRAM", "energy/physical_total_sram_nj",
           f"W, H are {report['energy/drive_head_share']:.0%} of this")
        nj("  + W, H from DRAM/HBM", "energy/physical_total_dram_nj")
        nj("  + W, H from seed", "energy/physical_total_seed_nj")

    if "energy/baseline_macs" in report:
        rows.append((baseline_name, "", f"{report['energy/baseline_macs']:.3g} MACs"))
        nj("  weights in SRAM", "energy/baseline_sram_nj")
        nj("  weights from DRAM/HBM", "energy/baseline_dram_nj")
        for source, label in (("sram", "W, H in SRAM"), ("seed", "W, H from seed")):
            key = f"energy/baseline_over_physical_{source}"
            if key in report:
                rows.append((f"  / physical, {label}", f"{report[key]:12.3g} x", ""))

    width = max(len(label) for label, _, _ in rows)
    rule = "-" * (width + 50)
    title = f"Energy per sample (batch {report['energy/batch_size']:.0f}, digital: {costs.name})"
    lines = [title, rule]
    lines += [f"{label.ljust(width)}  {value.rjust(15)}  {note}".rstrip() for label, value, note in rows]
    lines.append(rule)
    if physical:
        lines.append("oscillator-hardware figures are OscillatorHardware's assumptions, not measurements")
    return "\n".join(lines)


def override(obj, items: Iterable[str]):
    """A copy of dataclass `obj` with "field=value" strings applied, values parsed as floats.

    For command lines: `--hw cycles_per_unit_time=200 --hw coupler_power_w=1e-7`.
    """
    changes = {}
    fields = {f.name: f for f in dataclasses.fields(obj)}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or key not in fields:
            raise ValueError(f"expected field=value with field in {sorted(fields)}, got {item!r}")
        changes[key] = type(getattr(obj, key))(float(value))
    return dataclasses.replace(obj, **changes)


def add_energy_args(parser: argparse.ArgumentParser) -> None:
    """The flags every script that prints an energy report shares; read by report_from_args."""
    group = parser.add_argument_group("energy estimate")
    group.add_argument("--energy-costs", choices=sorted(DIGITAL_PRESETS), default="7nm-int8",
                       help="per-op and per-byte energies for the digital parts")
    group.add_argument("--energy-batch-size", type=int, default=1,
                       help="samples sharing each weight read")
    group.add_argument("--energy-baseline", default=None, metavar="DIMS",
                       help="reference MLP to compare against, e.g. 784,256,10; 'none' for no baseline")
    group.add_argument("--hw", action="append", default=[], metavar="FIELD=VALUE",
                       help="override an OscillatorHardware field; repeatable")


def report_from_args(
    model: nn.Module, args: argparse.Namespace, default_baseline: Iterable[int] | None = None
) -> tuple[dict[str, float], str]:
    """(energy_report, its formatted table) configured by add_energy_args' flags.

    `default_baseline` is the reference MLP's dims when --energy-baseline is not given.
    """
    costs = DIGITAL_PRESETS[args.energy_costs]
    spec = args.energy_baseline
    if spec is None:
        dims = list(default_baseline) if default_baseline is not None else None
    elif spec.lower() == "none":
        dims = None
    else:
        dims = [int(d) for d in spec.split(",")]
    baseline = mlp_workload(dims) if dims else None
    baseline_name = "MLP " + "-".join(map(str, dims)) if dims else "baseline"
    hw = override(OscillatorHardware(), args.hw)
    report = energy_report(model, costs, hw, baseline, args.energy_batch_size)
    return report, format_energy_report(report, costs, baseline_name)


def main() -> None:
    from pymoto.models import create_model

    parser = argparse.ArgumentParser(description="Energy report for a registered model at init.")
    parser.add_argument("model", help="registered model name, e.g. kuramoto_mnist")
    parser.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE",
                        help="config override, e.g. --set n=1024; repeatable")
    add_energy_args(parser)
    args = parser.parse_args()

    config_overrides = {}
    for item in args.set:
        key, _, value = item.partition("=")
        config_overrides[key] = float(value) if "." in value or "e" in value else int(value)
    model = create_model(args.model, **config_overrides)
    print(report_from_args(model, args)[1])


if __name__ == "__main__":
    main()

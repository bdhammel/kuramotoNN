# pymoto

A framework for experimenting with Kuramoto oscillator networks: networks
whose hidden state is a population of coupled phase oscillators, and whose
only trainable tensor is the coupling matrix `K`. See
[`../docs/kuramoto-oscillators.md`](../docs/kuramoto-oscillators.md) for the
theory, the diagnostics, and the design decisions.

The code follows the conventions of `torch.nn`, `timm` and Hugging Face
`transformers`. Each stage of the network is a small block you can build, test,
calibrate and swap on its own. Models are thin compositions of those blocks.

## Anatomy of the network

```
x (B, in_dim)
  │  drive     FrozenDrive         z = x @ W.T                                  W frozen, calibrated
  ▼
z (B, n)
  │  dynamics  KuramotoDynamics    theta_0 = 0;  num_steps Euler steps over [0, T] of
  │                                dtheta/dt = g*z + sum_j K_ij sin(theta_j - theta_i)
  │    └ coupling KuramotoCoupling the sum_j term; owns K                        K TRAINABLE
  ▼
theta (B, n)
  │  readout   PhaseReadout        theta - mean(theta) -> [sin, cos]           no state
  ▼
features (B, 2n)                   <- forward_features(x) stops here
  │  head      FrozenHead          logits = features @ H.T / tau              H frozen, tau calibrated
  ▼
logits (B, num_classes)            <- forward(x)
```

| component | block | state | Hugging Face analog | timm analog |
|---|---|---|---|---|
| drive | `FrozenDrive` | `W (n, in_dim)` buffer | embeddings | stem |
| dynamics | `KuramotoDynamics` | `g` buffer; `T`, `num_steps` | encoder | blocks: one block, weight-tied, applied `num_steps` times |
| coupling | `KuramotoCoupling` | **`K (n, n)` parameter**, `k_scale` buffer | the encoder layer's weights | the block's weights |
| readout | `PhaseReadout` | none | pooler | `global_pool` / `pre_logits` |
| head | `FrozenHead` | `H (C, 2n)`, `tau` buffers | classifier | `head` / `fc` |

## Building blocks: `pymoto.layers`

Every block follows the same conventions, taken from `torch.nn` and `timm.layers`:

- **The math is a plain function; the module owns the state.** For example,
  `coupling(theta, K_eff)` is the module-free form of `KuramotoCoupling`, and
  `readout_features(theta)` of `PhaseReadout`. The functions are testable on their own.
  `coupling_pairwise` is the literal reference form, and a test checks the fast
  form against it.
- **Sub-blocks are passed in, not built inside.** `KuramotoDynamics(coupling, T, num_steps)`
  takes the coupling module it integrates, so the coupling can be built, tested or
  replaced separately.
- **Keyword-only `generator`, `device`, `dtype`**, as on `nn.Linear`. Random state is
  always drawn on the CPU in float32 and then cast, so a seed gives the same init on
  CPU, MPS or CUDA, and in float32 or float64.
- **`reset_parameters(generator)`** re-draws a block's random state.
- **`calibrate_(batch)`** is a block's data-dependent init, done in place, as in ActNorm
  or LSUV. `FrozenDrive.calibrate_(x)` scales W so the drive has unit std, and
  `FrozenHead.calibrate_(features)` sets tau so the logits have unit std.
- **Frozen means buffer.** Only trainable tensors are `nn.Parameter`s, so
  `model.parameters()` is exactly what the optimizer should see. Nothing can make W,
  H, g or tau trainable by accident.
- **`extra_repr`** on every block, so `print(model)` shows the hyperparameters.

Building the network by hand from the blocks. This is the same thing
`KuramotoModel` does:

```python
import torch
from pymoto.layers import FrozenDrive, FrozenHead, KuramotoCoupling, KuramotoDynamics, PhaseReadout

gen = torch.Generator().manual_seed(0)
coupling = KuramotoCoupling(256, k_scale=1.0, generator=gen)      # K ~ N(0, 1/n), diag 0
drive = FrozenDrive(784, 256, generator=gen)                       # W ~ N(0, 1)
dynamics = KuramotoDynamics(coupling, T=1.0, num_steps=10)         # g = 1 until calibrated
readout = PhaseReadout()
head = FrozenHead(512, 10, generator=gen)                          # H ~ N(0, 1), tau = 1

drive.calibrate_(x_cal)
dynamics.g.fill_(1.0)
head.calibrate_(readout(dynamics(drive(x_cal))))

logits = head(readout(dynamics(drive(x))))
```

### Integrators: a step, applied num_steps times

`pymoto.layers.integrators` is generic over any autonomous `dx/dt = f(x)` and
knows nothing about oscillators. It separates the two things that vary
independently:

| | |
|---|---|
| `euler_step(f, x, h)`, `rk4_step(f, x, h)` | one update: the residual block |
| `rollout(step, f, x0, h, N)` | apply `step` N times and return the final state |
| `trajectory(step, f, x0, h, N)` | apply `step` N times and return all `N + 1` states |

`KuramotoDynamics.velocity_field(z)` binds one batch's drive and returns the
vector field `f`. It computes `K_eff` once rather than once per step. The
dynamics block is then just `rollout(euler_step, velocity_field(z), 0, h, N)`.
The model always uses `euler_step`. There is deliberately no solver setting,
and `rk4_step` is an evaluation tool (see the controls below).

## Training

- **Calibrate once, at init.** `pymoto.calibrate(model, x_cal, g)` runs the blocks'
  `calibrate_` steps in the order that matters, because each measures the output of
  the one before: drive (W), then g, then head (tau, measured through the calibrated
  drive and dynamics).
- **Optimize K only.** `torch.optim.Adam(model.parameters())` sees exactly
  `kuramoto.dynamics.coupling.K`. `pymoto.diagnostics.assert_only_K_is_trainable(model)`
  is the guard to call before training.
- **Gradient checkpointing.** Autograd keeps every Euler step's activations, so memory
  grows with `num_steps`. `model.set_grad_checkpointing()` (the timm name) makes the
  dynamics keep only each step's input and recompute the rest during backward, for one
  extra forward pass. Gradients are unchanged: tests check they are bit-identical, and
  that saved activations drop by more than 3x at 20 steps.
- **float64 runs.** `create_model(..., dtype=torch.float64)`, or `dtype=` on any block,
  gives the same init in double precision. This is the natural way to probe the
  float32-rounding effects discussed in `effective_coupling`'s docstring.
- **Monitoring.** `pymoto.diagnostics.compute_diagnostics(model, x_cal)` returns
  `rho`, the participation ratio, `h * lambda_max(J)` and the other diagnostics, and
  `warn_on_diagnostics` flags values out of range. Run it at init and after each epoch;
  K is trained, so the ODE's stiffness can drift.
- **Trajectories.** `model.kuramoto(x, output_trajectory=True).trajectory` is theta at
  every Euler step, shape `(num_steps + 1, B, n)`. This is HF's `output_hidden_states`
  and timm's `forward_intermediates`. `dynamics.forward_trajectory(z)` gives the same
  from the block.

## Evaluation: `pymoto.controls`

The controls make an accuracy attributable to K. Each one builds a **variant**
of the model with the same `forward(x) -> logits`, so the task scores it with
its own metric: test accuracy for MNIST, episode reward for CartPole.

| control | call | a result means |
|---|---|---|
| input severed | `with_num_steps(model, 0)` | features identical for every input, so this is the floor |
| reservoir baseline | `with_coupling(model, K_init)` | if trained K ≈ this, K learned nothing |
| solver transfer | `with_solver(model, rk4_step, refine=10)` | a large drop means K fit the Euler grid rather than a flow |
| linear probe | `linear_probe(model, train, test, device)` | a ceiling on the frozen features; the probe is never part of the model |

`with_num_steps` and `with_coupling` return modified copies. `with_solver`
wraps the model and shares its weights. See `experiments/mnist/eval.py` and
`experiments/cartpole/eval.py --controls` for both tasks.

## Energy estimates: `pymoto.energy`

Energy per sample, counted as events × energy per event, in two ways:

- **Digital:** MACs plus weight reads, from on-chip SRAM or off-chip DRAM/HBM.
  This is how an MLP baseline is costed, and how the Kuramoto network is costed
  when it is simulated with Euler steps.
- **Physical:** the network run as real oscillators.
  - K lives in the coupling fabric and is never read from memory.
  - `num_steps` does not exist; energy scales with `T`.
  - The terms are DAC writes that set each oscillator's drive, oscillator-cycles,
    coupler power, the wires carrying each oscillator's signal to its couplers, and
    the I/Q readout conversions.
  - The frozen `W` and `H` stay digital. They are costed with their weights read
    from SRAM, read from DRAM, or regenerated from their seed.

```python
from pymoto.energy import energy_report, format_energy_report, mlp_workload
report = energy_report(model, baseline=mlp_workload((784, 256, 10)))   # flat dict, nJ
print(format_energy_report(report, baseline_name="MLP 784-256-10"))
```

Two ratios in the report decide whether the physics is worth building:

- `sim_over_core`: the digital simulation's energy over the physical core's. It
  should be ≫ 1. A circuit that is cheaper to simulate than to run is not worth
  building.
- `drive_head_share`: the digital `W` and `H`'s share of the physical total. Near 1
  means the oscillators cannot help until `W` stops being read from memory
  (Amdahl's law).

The digital presets are published circuit figures: `7nm-int8` (Google, ISCA
2021, the default), `45nm-int8` and `45nm-fp32` (Horowitz, ISSCC 2014).

`OscillatorHardware`'s defaults are **placeholders, not measurements**: CMOS ring
oscillators at 1 GHz, with 100 carrier cycles per unit of model time. Override
them with `--hw field=value`.

Both experiments' `eval.py` and `train.py` print the report and log it (JSON /
wandb summary), and take `--energy-costs`, `--energy-batch-size`,
`--energy-baseline DIMS` and `--hw`. Without a checkpoint:

```sh
python -m pymoto.energy kuramoto_mnist --energy-baseline 784,256,10
python -m pymoto.energy kuramoto_mnist --set n=1024 --hw cycles_per_unit_time=300
```

## Model level

These are thin compositions of the blocks, following Hugging Face and timm:

- `KuramotoConfig` holds the architecture: `n`, `in_dim`, `num_classes`, `T`,
  `num_steps` and `k_scale`. Anything fitted to data, including K, W's scale, g and
  tau, lives in the state dict.
- `KuramotoModel` is the base model (drive → dynamics → readout) and returns
  `KuramotoModelOutput(drive, phases, features, trajectory)`.
- `KuramotoForClassification` adds the head: `forward_features`, `forward_head`,
  `get_classifier()`, `get_coupling()`, and `base_model_prefix = "kuramoto"`.
- `save_pretrained(dir)` / `from_pretrained(dir)` write and read `config.json` plus
  `pytorch_model.bin`.
- `create_model("kuramoto_mnist" | "kuramoto_cartpole", **overrides)` builds a
  registered variant, and `checkpoint_filter_fn` loads pre-pymoto flat checkpoints.

```
src/pymoto/
├── layers/              building blocks (above)
├── controls.py          evaluation variants and the linear probe
├── diagnostics.py       calibration diagnostics, spectral quantities, attribution guard
├── energy.py            energy per sample: digital simulation vs physical oscillators
├── configuration_utils.py, modeling_utils.py   PretrainedConfig, PreTrainedModel   [HF]
└── models/
    ├── _registry.py     register_model, create_model, list_models                  [timm]
    └── kuramoto/        configuration_kuramoto.py, modeling_kuramoto.py            [HF]
```

## Invariants

- **Bit-for-bit compatible with the pre-pymoto code.** A given seed produces the same
  weights, logits, K gradients (with or without checkpointing), diagnostics and control
  results as the old `experiments/mnist/model.py:KuramotoClassifier` and
  `experiments/cartpole/model.py:KuramotoPolicy`. The model builds the coupling first
  so the generator draws K, then W, then H, the same order as before.
- **No solver setting on the model.** It always integrates with Euler.

## Install and test

```sh
uv pip install -e "pymoto[test]"     # from the repo root
python -m pytest pymoto/tests
```

In this repo the venvs live in `.venv.nosync/` (symlinked as `.venv`). The repo is
in iCloud-synced `~/Documents`, and iCloud hides files inside ordinary dot-folders,
which makes Python skip an editable install's `.pth` file.

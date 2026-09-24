# Kuramoto-oscillator MNIST classifier

A frozen random projection, a coupled-oscillator ODE, and a frozen random
readout. **The coupling matrix `K` is the only trainable tensor in the
model.** See [`../../docs/kuramoto-oscillators.md`](../../docs/kuramoto-oscillators.md)
for the theory, the calibration diagnostics, the five controls, and the
design decisions behind this architecture.

```
x (B,784) --W--> z (B,256)        W frozen random, scale calibrated
theta_0 = 0
dtheta/dt = g*z + sum_j K_ij sin(theta_j - theta_i)      Euler, 10 steps, T = 1.0
theta -> theta - mean(theta)  ->  [sin, cos]  ->  (B,512)
logits = features @ H.T / tau                            H frozen random, no bias
```

The model itself is pymoto's `kuramoto_mnist`
(`create_model("kuramoto_mnist")`); see [`../../pymoto`](../../pymoto) for
its component breakdown. This directory holds only the experiment: data,
training loop, controls.

## Setup

```sh
uv pip install -r requirements.in   # includes pymoto, editable
wandb login          # runs default to --wandb-mode online
```

MNIST (~11 MB) downloads to `./data` on first run.

## 1. Calibration table

Calibrate `W`, `g` and `tau`, print the init diagnostics, stop.

```sh
python train.py --calibrate-only --wandb-mode disabled
```

## 2. Training

Adam on `K` alone, `lr=1e-3`, batch 128, 30 epochs, 5k held-out validation images.

```sh
python train.py --epochs 30 --seed 0
```

Writes `runs/best/` (best validation accuracy) and `runs/final/`. Each is a
pymoto `save_pretrained` directory (`config.json`, `pytorch_model.bin`) plus
`training_state.pt`, which holds `K_init` -- the random-`K` control needs it --
and the hparams and data statistics.

Per epoch, wandb receives train/val loss and accuracy plus `||K||_2`,
`||K||_F`, `h * lambda_max(J)`, participation ratio, wrapped-phase
fraction, `rho`, and the asymmetry `||K - K.T||_F / ||K||_F`.

Useful overrides: `--g`, `--k-scale`, `--num-steps`, `--n`, `--lr`,
`--device`, `--wandb-mode {online,offline,disabled}`.

## 3. Evaluate / the five controls

Run at the end of every training run and logged as wandb summary values.
Also standalone against any checkpoint:

```sh
python eval.py --checkpoint runs/final --json-out runs/controls.json
```

## 4. Energy estimate

Printed after the controls, written to `--json-out`, and logged to wandb by
`train.py`. It is the energy per sample, simulated digitally vs run as
physical oscillators, compared against an MLP of the same hidden width
(`784-n-10`). See `pymoto.energy` for the model and its assumptions. The ratios
to the MLP mean something only at matched accuracy, so pass
`--energy-baseline` for an MLP that reaches it.

```sh
python eval.py --checkpoint runs/final --energy-only             # no data, no controls
python eval.py --checkpoint runs/final --energy-only --hw cycles_per_unit_time=300
```

Single-file checkpoints written before the move to pymoto (`runs/*.pt`) still
load: their flat `K`/`W`/`H` state dicts are remapped by
`pymoto.checkpoint_filter_fn`.

## Files

| file | contents |
|---|---|
| `data.py` | download, global-scalar standardization, 55k/5k/10k split, loaders |
| `train.py` | `HParams`, training loop, wandb, per-epoch diagnostics |
| `eval.py` | the five controls scored on the test set, plus the energy estimate; standalone against a checkpoint |
| `utils.py` | seeding, device selection, checkpoint save/load |

The model, `calibrate`, the control variants (`pymoto.controls`) and the
diagnostics (participation ratio, spectral norms, the only-`K`-is-trainable
guard) come from `pymoto`.

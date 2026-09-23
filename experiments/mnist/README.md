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

## Setup

```sh
uv pip install -r requirements.in
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

Writes `runs/best.pt` (best validation accuracy) and `runs/final.pt`. Both
store `K_init`, which the random-`K` control needs.

Per epoch, wandb receives train/val loss and accuracy plus `||K||_2`,
`||K||_F`, `h * lambda_max(J)`, participation ratio, wrapped-phase
fraction, `rho`, and the asymmetry `||K - K.T||_F / ||K||_F`.

Useful overrides: `--g`, `--k-scale`, `--num-steps`, `--n`, `--lr`,
`--device`, `--wandb-mode {online,offline,disabled}`.

## 3. Evaluate / the five controls

Run at the end of every training run and logged as wandb summary values.
Also standalone against any checkpoint:

```sh
python eval.py --checkpoint runs/final.pt --json-out runs/controls.json
```

## Files

| file | contents |
|---|---|
| `data.py` | download, global-scalar standardization, 55k/5k/10k split, loaders |
| `model.py` | coupling term, Euler and RK4 rollouts, readout, classifier, calibration |
| `train.py` | `HParams`, training loop, wandb, per-epoch diagnostics |
| `eval.py` | the five controls, standalone against a checkpoint |
| `utils.py` | seeding, diagnostics, participation ratio, spectral norms, checkpoints |

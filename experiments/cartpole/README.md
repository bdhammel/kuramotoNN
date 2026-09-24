# CartPole with REINFORCE

Two interchangeable policies, both trained with REINFORCE (vanilla policy
gradient) on `CartPole-v1` via `gymnasium`:

- `mlp` (default): `Linear(4, 128) -> ReLU -> Linear(128, 2)`.
- `kuramoto`: a Kuramoto-oscillator policy, pymoto's `kuramoto_cartpole` --
  the same architecture as the MNIST classifier (see [`../../pymoto`](../../pymoto)).
  A frozen random drive projects the observation into 64 oscillators by
  default (`--n-oscillators`), a trainable coupling
  matrix `K` evolves their phases for 10 Euler steps (`K` is the *only*
  trainable tensor), and a frozen random head reads out mean-relative
  `[sin, cos]` features into action logits. Before training starts, it's
  calibrated (W scale, drive gain `g`, logit temperature `tau`) on states
  gathered from a random-action rollout, the same idea as the MNIST
  calibration step. In testing, `--seed 1` (the default) solves CartPole in
  well under 200 episodes with this policy -- faster than the MLP.

See [`../../docs/kuramoto-oscillators.md`](../../docs/kuramoto-oscillators.md)
for the theory behind the `kuramoto` policy's architecture.

## Setup

```sh
uv pip install -r requirements.in   # includes pymoto, editable
```

## Train

```sh
wandb login             # runs default to --wandb-mode online
python train.py --episodes 1000
python train.py --policy kuramoto
python train.py --wandb-mode disabled   # skip wandb
```

Trains until the running average reward crosses 475 (CartPole-v1's "solved"
threshold, max episode length 500) or the episode budget runs out. Logs
per-episode reward, length, running reward, and loss to wandb (project
`cartpole-reinforce`), plus a `solved_at_episode` / `final_running_reward`
summary. Saves weights to `checkpoint.pt`.

## Evaluate

Runs the greedy (argmax) policy for N episodes and prints summary stats to
the console: mean/median/std reward, min/max, and success rate (fraction of
episodes hitting the max 500-step episode length).

```sh
python eval.py --episodes 100
python eval.py --render   # watch it play
python eval.py --controls # kuramoto only: also score the control variants
```

`--controls` scores pymoto's controls by mean greedy reward: `num_steps = 0`
(input severed), random `K` (the exact init `K`, the reservoir baseline), and
RK4 on a 10x finer grid (solver transfer). Checkpoints from `train.py` store
`K_init`. For older checkpoints it is regenerated from the seed, which is exact
because `K` is the first thing drawn after `torch.manual_seed`.

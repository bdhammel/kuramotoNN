# InvertedDoublePendulum with REINFORCE

Two interchangeable policies, both trained with REINFORCE (vanilla policy
gradient) on `InvertedDoublePendulum-v5` (MuJoCo) via `gymnasium`:

- `mlp` (default): `Linear(9, 128) -> ReLU -> Linear(128, 1)`, the mean of a
  diagonal Gaussian over the 1-dim continuous action.
- `kuramoto`: a Kuramoto-oscillator policy, pymoto's
  `kuramoto_inverted_double_pendulum` -- the same architecture as the CartPole
  and MNIST models (see [`../../pymoto`](../../pymoto)). A frozen random drive
  projects the 9-dim observation into 64 oscillators by default
  (`--n-oscillators`), a trainable coupling matrix `K` evolves their phases
  for 10 Euler steps (`K` is the *only* trainable tensor), and a frozen
  random head reads out mean-relative `[sin, cos]` features into a single
  action mean. Calibrated the same way as the CartPole policy, on states from
  a random-action rollout.

Unlike CartPole's discrete action, InvertedDoublePendulum's action is a
continuous force in `[-1, 1]`. Both mean-nets are wrapped in `GaussianPolicy`
(`model.py`), which adds one learnable, state-independent `log_std` and turns
the mean-net's output into a `Normal(mean, std)` policy: `mean = tanh(mean_net(x))`
keeps the mean in bounds, but sampled actions can still land outside
`[-1, 1]` (`std` isn't squashed), so they're clipped before stepping the env.
`log_std` is REINFORCE's only other trainable parameter besides the mean-net
(or `K`, for `kuramoto`).

This is a substantially harder control problem than CartPole: continuous
actions, a 9-dim observation, 1000-step episodes, and a much higher "solved"
bar (mean reward >= 9100, vs. CartPole's 475).

See [`../../docs/kuramoto-oscillators.md`](../../docs/kuramoto-oscillators.md)
for the theory behind the `kuramoto` policy's architecture.

## Setup

```sh
uv pip install -r requirements.in   # includes pymoto (editable) and mujoco
```

## Train

```sh
wandb login             # runs default to --wandb-mode online
python train.py --episodes 3000
python train.py --policy kuramoto
python train.py --wandb-mode disabled   # skip wandb
```

Trains until the running average reward crosses 9100
(`InvertedDoublePendulum-v5`'s `reward_threshold`, max episode length 1000)
or the episode budget runs out. Logs per-episode reward, length, running
reward, loss, and the policy's current `std` to wandb (project
`inverted-double-pendulum-reinforce`), plus a `solved_at_episode` /
`final_running_reward` summary. Saves weights to `checkpoint.pt`.

## Evaluate

Runs the greedy (mean-action, clipped to `[-1, 1]`) policy for N episodes and
prints summary stats to the console: mean/median/std reward, min/max, and
success rate (fraction of episodes hitting the 9100 reward threshold).

```sh
python eval.py --episodes 20
python eval.py --render   # watch it balance
python eval.py --controls # kuramoto only: also score the control variants
```

`--controls` scores pymoto's controls by mean greedy reward: `num_steps = 0`
(input severed), random `K` (the exact init `K`, the reservoir baseline), and
RK4 on a 10x finer grid (solver transfer). Each control variant rewraps the
varied mean-net in a fresh `GaussianPolicy` sharing the trained `log_std`, so
only the mean-net changes. Checkpoints from `train.py` store `K_init`. For
older checkpoints it is regenerated from the seed, which is exact because `K`
is the first thing drawn after `torch.manual_seed`.

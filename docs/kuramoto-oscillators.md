# Kuramoto oscillators as a learned computational substrate

The idea explored across the experiments in this repo: take a network of
coupled oscillators, make the coupling matrix `K` the *only* trainable
tensor, and see whether training `K` produces a genuine learned dynamical
system -- as opposed to a fixed random reservoir that happens to be
readable.

```
x (B, in) --W--> z (B, n)        W frozen random, scale calibrated
theta_0 = 0
dtheta/dt = g*z + sum_j K_ij sin(theta_j - theta_i)      Euler, N steps, T = 1.0
theta -> theta - mean(theta)  ->  [sin, cos]  ->  (B, 2n)
logits = features @ H.T / tau                             H frozen random, no bias
```

The input enters only as a constant drive term and the initial phases are
exactly zero, so with zero integration steps the input has no path to the
output at all. Any accuracy above chance is attributable to `K` and to the
dynamics, and to nothing else.

## Calibration table

Before training, `W`, `g`, and `tau` are calibrated and the following
diagnostics are checked:

| quantity | target | meaning if violated |
|---|---|---|
| `std_i(theta - theta_bar)` | ≈ 1 rad | adjust `--g` |
| `frac(\|theta - theta_bar\| > pi)` | < 1% | `--g` too large, phases aliasing |
| `Var(theta_bar) / Var(theta - theta_bar)` | ≪ 1 | standardization is broken |
| `rho = ||coupling|| / ||g*z||` | 0.3 – 2 | adjust `--k-scale` |
| participation ratio of the features | ≫ 10 | features rank-collapsed |
| `h * lambda_max(J)`, `J = K_eff - diag(K_eff @ 1)` | ≪ 1 | step size unsafe |

Out-of-range values print an explicit `WARNING` line. The participation
ratio is the one to look at first: a frozen random projection samples the
feature covariance roughly democratically, so too few well-populated
directions means no amount of training `K` will separate the output
classes.

## The five controls

Run at the end of training and available standalone against any checkpoint:

1. **Chance** -- the majority-class / uniform-random floor.
2. **`num_steps = 0`** -- features are input-independent by construction
   (phases never leave `theta_0 = 0`), so the reported accuracy is the
   majority-class floor rather than exactly the naive chance rate.
3. **Random-`K`** -- the same model evaluated with `K` frozen at its init
   value. `K` is the only trainable tensor, so if trained-`K` ≈ random-`K`
   the system is a reservoir rather than a learned dynamical system. This
   is the single most important number in the project.
4. **Linear probe** -- a *trainable* head fit on the frozen features, as a
   pure measurement. Upper-bounds what the frozen head can reach and
   separates "the features are bad" from "the frozen head cannot read good
   features." It is discarded immediately and is never part of the model.
5. **Solver transfer** -- trained with Euler at `num_steps`, evaluated with
   RK4 at a much finer resolution over the same `T`. A large drop means the
   model overfit the discretization rather than learning a flow.

## Design decisions worth knowing about

**Standardization is a single global scalar**, not per-feature, where
per-feature values can be identically constant in the training data (e.g.
MNIST's border pixels). A per-feature std is exactly 0 there; an epsilon
floor avoids the NaN but then multiplies pure-noise test-time deviations by
`1/eps`, swamping the drive with directions that carry no signal.

**The coupling term is computed as two matmuls.**
`sum_j K_ij sin(theta_j - theta_i)` is expanded via
`sin(a-b) = sin a cos b - cos a sin b` into
`cos(theta) * (sin(theta) @ K.T) - sin(theta) * (cos(theta) @ K.T)`, which
costs two `(B,n)@(n,n)` products instead of an `(B,n,n)` per-sample outer
difference.

**`K`'s diagonal is left alone**, and `K_eff` is just `k_scale * K`. The
`j = i` term of the coupling is identically zero, so the diagonal is a null
direction of the model. It is zeroed at init for a clean starting state,
then allowed to drift under training -- Adam normalizes by the gradient's
own RMS, so the ~1e-7 autograd residue on the diagonal becomes a near-full
step and `K_ii` random-walks out to roughly the scale of a real entry. Two
consequences:

- `h * lambda_max(J)` is **unaffected** -- `J = K_eff - diag(K_eff @ 1)` is
  exactly invariant to the diagonal, since adding `d` to `K_ii` adds `d` to
  both the entry and its row sum.
- The **norms are not** invariant, and that one matters: a drifted diagonal
  pulls `||K - K.T||_F / ||K||_F` down, which is exactly the signature read
  as evidence of a learned gradient flow. Diagnostics therefore strip the
  diagonal before measuring `||K||_2`, `||K||_F`, and the asymmetry ratio.

The null-direction argument is specific to this coupling term. Under a
Sakaguchi lag `sin(theta_j - theta_i - phi)` the `j = i` term is
`-sin(phi) != 0` and the diagonal becomes a live self-drive -- which is why
that convention exists elsewhere in the literature.

**`k_scale` is a buffer, not a trainable parameter.** It sets the balance
between the two terms of `dtheta/dt = g*z + k_scale * coupling`, and is the
knob the `rho` diagnostic tells you to turn. It picks the *regime*: too
small and the oscillators barely interact, so `theta ≈ g*z*T` and the
readout degenerates into a fixed pointwise nonlinearity on a random
projection with no dynamics for `K` to shape; too large and the coupling
synchronizes the population, phases collapse toward a common value, and the
features rank-collapse -- visible immediately as a low participation ratio.

It is deliberately not trainable for two independent reasons:

1. **It would be redundant.** `K_eff = k_scale * K` and `K` is fully
   trainable, so any value `k_scale` could learn is already reachable by
   rescaling `K`. It adds exactly zero expressive capacity.
2. **It would cost the attribution.** The claim these experiments exist to
   support is that `K` is the only trainable tensor, so accuracy above
   chance is attributable to `K` and the dynamics alone. One extra trained
   scalar forfeits that for nothing.

Keeping it out of `K`'s initializer, rather than folding it into the init
std, is what makes a `k_scale` sweep meaningful: at a fixed seed every
value gets the *same* random `K`, varying only its strength.

**The Euler loop is written by hand.** No `torchdiffeq`, no adjoint. Each
step is `theta = theta + h * velocity(theta)` -- a residual block with tied
weights, making the rollout a weight-tied ResNet of depth `num_steps`.

Deliberately excluded, and not behind flags: SHIL / sub-harmonic injection
locking, Sakaguchi phase lag, trainable `omega`, random initial phases,
phase wrapping during integration, and symmetrizing `K`.

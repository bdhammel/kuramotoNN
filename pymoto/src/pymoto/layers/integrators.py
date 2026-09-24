"""Fixed-step ODE integrators for an autonomous system dx/dt = f(x).

Generic: they know nothing about oscillators. Split into the two pieces that
vary independently:

    euler_step, rk4_step    one update x -> x', given the vector field f and step h
    rollout, trajectory     apply a step num_steps times; final state or every state

A Kuramoto rollout is `rollout(euler_step, dynamics.velocity_field(z), 0, h, N)`.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

VectorField = Callable[[Tensor], Tensor]
Step = Callable[[VectorField, Tensor, float], Tensor]


def euler_step(f: VectorField, x: Tensor, h: float) -> Tensor:
    """x -> x + h * f(x). Explicit Euler.

    Written out by hand rather than handed to an ODE library, so that the
    structure stays visible: each step is a residual update, and because f is the
    same at every step, a rollout is a weight-tied ResNet of depth num_steps.
    """
    return x + h * f(x)


def rk4_step(f: VectorField, x: Tensor, h: float) -> Tensor:
    """x -> x'. Classical RK4.

    Evaluation only for Kuramoto models. The model is always integrated with
    Euler; this exists so an evaluation (pymoto.controls.solver_transfer) can ask
    whether the trained K describes a flow or merely a fixed discretization. It is
    not a solver toggle on the model.
    """
    k1 = f(x)
    k2 = f(x + 0.5 * h * k1)
    k3 = f(x + 0.5 * h * k2)
    k4 = f(x + h * k3)
    return x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _apply_step(step: Step, f: VectorField, x: Tensor, h: float, grad_checkpointing: bool) -> Tensor:
    if grad_checkpointing and torch.is_grad_enabled():
        # Non-reentrant: gradients still reach tensors f closes over (K_eff),
        # which are not explicit inputs of the checkpointed call.
        return checkpoint(step, f, x, h, use_reentrant=False)
    return step(f, x, h)


def rollout(
    step: Step,
    f: VectorField,
    x0: Tensor,
    h: float,
    num_steps: int,
    *,
    grad_checkpointing: bool = False,
) -> Tensor:
    """x0 -> x at t = num_steps * h.

    grad_checkpointing: keep only each step's input for backward and recompute the
        step's internals during backward. Activation memory goes from
        O(num_steps * per-step activations) to O(num_steps * |x|), for one extra
        forward. Results and gradients are unchanged.
    """
    assert num_steps >= 0
    x = x0
    for _ in range(num_steps):
        x = _apply_step(step, f, x, h, grad_checkpointing)
    return x


def trajectory(
    step: Step,
    f: VectorField,
    x0: Tensor,
    h: float,
    num_steps: int,
    *,
    grad_checkpointing: bool = False,
) -> Tensor:
    """x0 -> (num_steps + 1, *x0.shape): the state at t = 0, h, 2h, ..., num_steps * h.

    The last entry is exactly what `rollout` returns with the same arguments.
    """
    assert num_steps >= 0
    states = [x0]
    for _ in range(num_steps):
        states.append(_apply_step(step, f, states[-1], h, grad_checkpointing))
    return torch.stack(states)

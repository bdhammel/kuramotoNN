"""Reusable building blocks, in the style of timm.layers.

Each block is a plain function (the math, testable in isolation) plus an nn.Module
that owns whatever state the function needs. Models in pymoto.models are
compositions of these; nothing here knows which model it belongs to.

    FrozenDrive         x -> z              W frozen, calibrate_() sets its scale
    KuramotoDynamics    z -> theta(T)       Euler over [0, T], theta_0 = 0
      KuramotoCoupling  theta -> coupling   K trainable, k_scale fixed; passed in
    PhaseReadout        theta -> features   mean-relative [sin, cos]
    FrozenHead          features -> logits  H frozen, calibrate_() sets tau

    euler_step, rk4_step, rollout, trajectory   generic fixed-step integration

Conventions shared by every module here, following torch.nn and timm:
    - sub-blocks are passed in, not built internally (KuramotoDynamics(coupling, ...))
    - keyword-only `generator`, `device`, `dtype`; initial weights are drawn on the
      CPU in float32 so a seed gives the same init on any device or dtype
    - reset_parameters(generator) re-draws the block's random state
    - calibrate_(batch) is the block's data-dependent init, in place
    - frozen tensors are buffers, so only trainable tensors appear in parameters()
"""

from pymoto.layers.coupling import KuramotoCoupling, coupling, coupling_pairwise, effective_coupling
from pymoto.layers.drive import FrozenDrive, TrainableDrive
from pymoto.layers.dynamics import KuramotoDynamics, velocity
from pymoto.layers.head import FrozenHead, TrainableHead
from pymoto.layers.integrators import Step, VectorField, euler_step, rk4_step, rollout, trajectory
from pymoto.layers.readout import PhaseReadout, readout_features

__all__ = [
    "FrozenDrive",
    "FrozenHead",
    "KuramotoCoupling",
    "KuramotoDynamics",
    "PhaseReadout",
    "Step",
    "TrainableDrive",
    "TrainableHead",
    "VectorField",
    "coupling",
    "coupling_pairwise",
    "effective_coupling",
    "euler_step",
    "readout_features",
    "rk4_step",
    "rollout",
    "trajectory",
    "velocity",
]

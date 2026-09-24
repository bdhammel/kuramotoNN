"""The controls: model variants and measurements that make an accuracy attributable to K.

Every reported number should come with these. Each one is task-agnostic: it
either builds a *variant* of the model -- something with the same forward(x) ->
logits, so the task scores it with its own metric (test accuracy, episode reward)
-- or measures the frozen features directly.

    with_num_steps(model, 0)          no integration: the input is severed from the
                                      output by construction, so this is the floor
    with_coupling(model, K_init)      the untrained coupling: the reservoir baseline,
                                      and the single most important comparison --
                                      if trained K ~= random K, K learned nothing
    linear_probe(model, ...)          a trainable head on the frozen features, as a
                                      measurement only: separates "the features are
                                      bad" from "the frozen head cannot read them"
    with_solver(model, rk4_step, r)   the same ODE on a finer grid with a better
                                      integrator: a large drop means K fit the Euler
                                      discretization rather than learning a flow
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from pymoto.layers.integrators import Step, rk4_step, rollout
from pymoto.modeling_utils import PreTrainedModel
from pymoto.models.kuramoto import KuramotoForClassification


def with_num_steps(model: KuramotoForClassification, num_steps: int) -> KuramotoForClassification:
    """A copy of `model` integrated with `num_steps` Euler steps over the same T.

    At num_steps = 0 the phases stay at theta_0 = 0, the mean-relative readout maps
    that to 0, and the features are [sin 0, cos 0] = [0, 1] for every input.
    """
    clone = copy.deepcopy(model)
    clone.kuramoto.dynamics.num_steps = num_steps
    config = dataclasses.replace(clone.config, num_steps=num_steps)
    for module in clone.modules():
        if isinstance(module, PreTrainedModel):
            module.config = config
    return clone


@torch.no_grad()
def with_coupling(model: KuramotoForClassification, K: Tensor) -> KuramotoForClassification:
    """A copy of `model` with coupling matrix K, everything else identical."""
    clone = copy.deepcopy(model)
    clone_K = clone.get_coupling().K
    clone_K.copy_(K.to(clone_K.device))
    return clone


class SolverTransfer(nn.Module):
    """(B, in_dim) -> logits: `model` with its ODE integrated by `step` at `refine` x the steps.

    Same drive, same K, same g, same total time T; only the integrator and grid
    change. Evaluation only -- the model itself is always integrated with Euler.
    """

    def __init__(self, model: KuramotoForClassification, step: Step = rk4_step, refine: int = 10) -> None:
        super().__init__()
        self.model = model
        self.step = step
        self.refine = refine

    @property
    def num_steps(self) -> int:
        return self.model.kuramoto.dynamics.num_steps * self.refine

    def forward_features(self, x: Tensor) -> Tensor:
        kuramoto = self.model.kuramoto
        dynamics = kuramoto.dynamics
        z = kuramoto.drive(x)
        steps = self.num_steps
        # theta_0 = 0, as in KuramotoDynamics.forward.
        theta = rollout(self.step, dynamics.velocity_field(z), torch.zeros_like(z), dynamics.T / steps, steps)
        return kuramoto.readout(theta)

    def forward(self, x: Tensor) -> Tensor:
        return self.model.forward_head(self.forward_features(x))

    def extra_repr(self) -> str:
        return f"step={getattr(self.step, '__name__', self.step)}, refine={self.refine}, num_steps={self.num_steps}"


def with_solver(model: KuramotoForClassification, step: Step = rk4_step, refine: int = 10) -> SolverTransfer:
    """`model` evaluated with a different integrator on a `refine` x finer grid. Shares weights."""
    return SolverTransfer(model, step=step, refine=refine)


@torch.no_grad()
def collect_features(
    model: KuramotoForClassification,
    loader: Iterable[tuple[Tensor, Tensor]],
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """-> ((N, 2n) frozen features, (N,) labels), both on `device`."""
    model.eval()
    feats, labels = [], []
    for x, y in loader:
        x = x.to(device)
        feats.append(model.forward_features(x))
        labels.append(y.to(device))
    return torch.cat(feats), torch.cat(labels)


def linear_probe(
    model: KuramotoForClassification,
    train_loader: Iterable[tuple[Tensor, Tensor]],
    test_loader: Iterable[tuple[Tensor, Tensor]],
    device: torch.device,
    epochs: int = 40,
    lr: float = 1e-2,
    batch_size: int = 512,
) -> dict[str, float]:
    """Fit a trainable 2n -> num_classes head on the frozen features. A measurement, never the model.

    This upper-bounds what the frozen random head can reach and separates "the
    features are bad" from "the frozen head cannot read good features". The probe
    is discarded; it is never attached to the classifier.
    """
    x_train, y_train = collect_features(model, train_loader, device)
    x_test, y_test = collect_features(model, test_loader, device)

    probe = nn.Linear(x_train.shape[1], model.num_classes).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    n = x_train.shape[0]

    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            loss = F.cross_entropy(probe(x_train[idx]), y_train[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    with torch.no_grad():
        train_acc = float((probe(x_train).argmax(-1) == y_train).float().mean())
        test_acc = float((probe(x_test).argmax(-1) == y_test).float().mean())
    return {"train_acc": train_acc, "test_acc": test_acc}

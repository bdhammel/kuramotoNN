"""The five controls. Every reported accuracy must be accompanied by all of them.

Run standalone against a checkpoint:

    python eval.py --checkpoint runs/final.pt

or import `run_controls` from train.py to log the same numbers into the wandb run
that produced the checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import json
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

from data import NUM_CLASSES, load_mnist
from model import KuramotoClassifier, rk4_rollout
from utils import load_checkpoint, pick_device, set_seed

CHANCE: float = 1.0 / NUM_CLASSES

# theta -> features -> logits, given the drive. Defaults to the model's own Euler
# rollout; the controls swap in a different integrator or step count.
PhasesFn = Callable[[KuramotoClassifier, Tensor], Tensor]


@torch.no_grad()
def evaluate(
    model: KuramotoClassifier,
    loader: DataLoader,
    device: torch.device,
    phases_fn: PhasesFn | None = None,
) -> tuple[float, float]:
    """Return (mean cross-entropy, accuracy) over a loader."""
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        z = model.drive(x)
        theta = model.integrate(z) if phases_fn is None else phases_fn(model, z)
        logits = model.classify(model.readout(theta))
        total_loss += float(F.cross_entropy(logits, y, reduction="sum"))
        correct += int((logits.argmax(dim=-1) == y).sum())
        count += y.numel()
    return total_loss / count, correct / count


@torch.no_grad()
def collect_features(
    model: KuramotoClassifier, loader: DataLoader, device: torch.device
) -> tuple[Tensor, Tensor]:
    """-> ((N, 2n) features, (N,) labels), both on `device`."""
    model.eval()
    feats, labels = [], []
    for x, y in loader:
        x = x.to(device)
        feats.append(model.readout(model.integrate(model.drive(x))))
        labels.append(y.to(device))
    return torch.cat(feats), torch.cat(labels)


# --------------------------------------------------------------------------
# Control 2: num_steps = 0
# --------------------------------------------------------------------------


@torch.no_grad()
def control_zero_steps(
    model: KuramotoClassifier, loader: DataLoader, device: torch.device
) -> dict[str, float]:
    """Correctness assertion, not a result: with no integration the input is severed.

    At num_steps = 0 the phases stay at theta_0 = 0, the mean-relative transform
    maps that to 0, and the features are [sin 0, cos 0] = [0, 1] for every input.

    A NOTE ON THE SPEC: this makes the features input-independent, which makes the
    *prediction* a single constant class. The resulting accuracy is therefore that
    class's frequency in the test set, not exactly 10% -- on MNIST's test split the
    per-class frequencies run from 8.92% (class 5) to 11.35% (class 1). "Exactly
    10%" would require the head to break ties uniformly at random, which would make
    the control noisy without making it stronger. The invariant that genuinely
    holds by construction is that the features carry zero information about the
    input, so that is what is asserted here; the accuracy is reported alongside it
    as the majority-class floor.
    """
    model.eval()
    correct, count = 0, 0
    preds_seen: set[int] = set()
    reference: Tensor | None = None
    max_feature_spread = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        features = model.readout(model.integrate(model.drive(x), num_steps=0))
        if reference is None:
            reference = features[0].clone()
        max_feature_spread = max(
            max_feature_spread, float((features - reference).abs().max())
        )
        preds = model.classify(features).argmax(dim=-1)
        preds_seen.update(preds.unique().tolist())
        correct += int((preds == y).sum())
        count += y.numel()

    assert max_feature_spread < 1e-5, (
        f"num_steps=0 features vary across inputs by up to {max_feature_spread:.3g}; "
        "the input has a path to the readout that does not go through the dynamics"
    )
    assert len(preds_seen) == 1, (
        f"num_steps=0 predicted {len(preds_seen)} distinct classes from constant features"
    )
    return {
        "acc": correct / count,
        "feature_spread": max_feature_spread,
        "predicted_class": float(next(iter(preds_seen))),
    }


# --------------------------------------------------------------------------
# Control 3: random K
# --------------------------------------------------------------------------


def model_with_K(model: KuramotoClassifier, K: Tensor) -> KuramotoClassifier:
    """Deep-copy the model with a different coupling matrix, everything else identical."""
    clone = copy.deepcopy(model)
    with torch.no_grad():
        clone.K.copy_(K.to(clone.K.device))
    return clone


# --------------------------------------------------------------------------
# Control 4: linear probe
# --------------------------------------------------------------------------


def linear_probe(
    model: KuramotoClassifier,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = 40,
    lr: float = 1e-2,
    batch_size: int = 512,
) -> dict[str, float]:
    """Fit a trainable 2n -> 10 head on the frozen features. A measurement, never the model.

    This upper-bounds what the frozen random head can reach and separates "the
    features are bad" from "the frozen head cannot read good features". The probe
    is discarded; it is never attached to the classifier.
    """
    x_train, y_train = collect_features(model, train_loader, device)
    x_test, y_test = collect_features(model, test_loader, device)

    probe = nn.Linear(x_train.shape[1], NUM_CLASSES).to(device)
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


# --------------------------------------------------------------------------
# Control 5: solver transfer
# --------------------------------------------------------------------------


def rk4_phases_fn(refine: int) -> PhasesFn:
    """Integrate the same ODE over the same T with RK4 at `refine` x the step count.

    A large accuracy drop against the Euler number means the trained K describes a
    particular discretization rather than a flow.
    """

    def phases(model: KuramotoClassifier, z: Tensor) -> Tensor:
        steps = model.num_steps * refine
        return rk4_rollout(z, model.K_eff(), model.g, model.T / steps, steps)

    return phases


# --------------------------------------------------------------------------


def run_controls(
    model: KuramotoClassifier,
    K_init: Tensor,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    rk4_refine: int = 10,
    probe_epochs: int = 40,
) -> dict[str, float]:
    """All five controls. Returns a flat dict suitable for wandb summary values."""
    results: dict[str, float] = {}

    # 1. Chance.
    results["control/chance"] = CHANCE

    # The headline number the controls exist to contextualize.
    _, trained_acc = evaluate(model, test_loader, device)
    results["control/trained_K_acc"] = trained_acc

    # 2. num_steps = 0.
    zero = control_zero_steps(model, test_loader, device)
    results["control/zero_steps_acc"] = zero["acc"]
    results["control/zero_steps_feature_spread"] = zero["feature_spread"]

    # 3. Random-K control. The single most important number in the project: if this
    #    matches trained-K, the system is a reservoir, not a learned dynamical system.
    _, random_acc = evaluate(model_with_K(model, K_init), test_loader, device)
    results["control/random_K_acc"] = random_acc
    results["control/lift_over_random_K"] = trained_acc - random_acc

    # 4. Linear probe on the frozen features.
    probe = linear_probe(model, train_loader, test_loader, device, epochs=probe_epochs)
    results["control/linear_probe_test_acc"] = probe["test_acc"]
    results["control/linear_probe_train_acc"] = probe["train_acc"]

    # 5. Solver transfer: Euler at num_steps vs RK4 at refine * num_steps.
    _, rk4_acc = evaluate(model, test_loader, device, phases_fn=rk4_phases_fn(rk4_refine))
    results["control/rk4_acc"] = rk4_acc
    results["control/solver_transfer_drop"] = trained_acc - rk4_acc

    return results


def format_controls(results: dict[str, float], rk4_refine: int, num_steps: int) -> str:
    """Render the control table for stdout."""
    rows = [
        ("1. chance", results["control/chance"], "1 / 10 by definition"),
        (
            "2. num_steps = 0",
            results["control/zero_steps_acc"],
            "constant features asserted; = majority-class floor",
        ),
        (
            "3. random K (init)",
            results["control/random_K_acc"],
            "reservoir baseline",
        ),
        (
            "   trained K",
            results["control/trained_K_acc"],
            f"lift {results['control/lift_over_random_K']:+.4f} over random K",
        ),
        (
            "4. linear probe",
            results["control/linear_probe_test_acc"],
            "diagnostic ceiling on the frozen features",
        ),
        (
            f"5. RK4 @ {num_steps * rk4_refine} steps",
            results["control/rk4_acc"],
            f"drop {results['control/solver_transfer_drop']:+.4f} vs Euler @ {num_steps}",
        ),
    ]
    width = max(len(label) for label, _, _ in rows)
    lines = ["Controls (test set)", "-" * (width + 46)]
    for label, value, note in rows:
        lines.append(f"{label.ljust(width)}  {value:8.4f}  {note}")
    lines.append("-" * (width + 46))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the five controls against a checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--rk4-refine", type=int, default=10)
    parser.add_argument("--probe-epochs", type=int, default=40)
    parser.add_argument("--json-out", default=None, help="optional path for the raw numbers")
    args = parser.parse_args()

    device = pick_device(args.device)
    model, ckpt = load_checkpoint(args.checkpoint, device)
    hp = ckpt["hparams"]
    set_seed(hp["seed"])

    data = load_mnist(
        root=args.data_root,
        batch_size=hp["batch_size"],
        val_size=hp["val_size"],
        seed=hp["seed"],
        calibration_size=hp["calibration_size"],
    )

    results = run_controls(
        model,
        ckpt["K_init"],
        data.train,
        data.test,
        device,
        rk4_refine=args.rk4_refine,
        probe_epochs=args.probe_epochs,
    )
    print(format_controls(results, args.rk4_refine, model.num_steps))

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()

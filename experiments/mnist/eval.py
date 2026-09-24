"""The five controls on MNIST. Every reported accuracy must be accompanied by all of them.

Run standalone against a checkpoint directory (or a pre-pymoto runs/*.pt file):

    python eval.py --checkpoint runs/final
    python eval.py --checkpoint runs/final --energy-only     # just the energy estimate

or import `run_controls` from train.py to log the same numbers into the wandb run
that produced the checkpoint. The controls themselves live in pymoto.controls;
this file scores them on the MNIST test set.
"""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

from pymoto import KuramotoForClassification
from pymoto.controls import linear_probe, with_coupling, with_num_steps, with_solver
from pymoto.energy import add_energy_args, report_from_args
from pymoto.layers import rk4_step

from data import NUM_CLASSES, load_mnist
from utils import load_checkpoint, pick_device, set_seed

CHANCE: float = 1.0 / NUM_CLASSES


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    """Return (mean cross-entropy, accuracy) over a loader, for anything mapping x -> logits."""
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += float(F.cross_entropy(logits, y, reduction="sum"))
        correct += int((logits.argmax(dim=-1) == y).sum())
        count += y.numel()
    return total_loss / count, correct / count


@torch.no_grad()
def control_zero_steps(
    model: KuramotoForClassification, loader: DataLoader, device: torch.device
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
    zero = with_num_steps(model, 0)
    zero.eval()
    correct, count = 0, 0
    preds_seen: set[int] = set()
    reference: Tensor | None = None
    max_feature_spread = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        features = zero.forward_features(x)
        if reference is None:
            reference = features[0].clone()
        max_feature_spread = max(
            max_feature_spread, float((features - reference).abs().max())
        )
        preds = zero.forward_head(features).argmax(dim=-1)
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


def run_controls(
    model: KuramotoForClassification,
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
    _, random_acc = evaluate(with_coupling(model, K_init), test_loader, device)
    results["control/random_K_acc"] = random_acc
    results["control/lift_over_random_K"] = trained_acc - random_acc

    # 4. Linear probe on the frozen features.
    probe = linear_probe(model, train_loader, test_loader, device, epochs=probe_epochs)
    results["control/linear_probe_test_acc"] = probe["test_acc"]
    results["control/linear_probe_train_acc"] = probe["train_acc"]

    # 5. Solver transfer: Euler at num_steps vs RK4 at refine * num_steps.
    _, rk4_acc = evaluate(with_solver(model, rk4_step, refine=rk4_refine), test_loader, device)
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


def run_energy(model: KuramotoForClassification, args: argparse.Namespace) -> tuple[dict[str, float], str]:
    """Energy per sample (pymoto.energy), against an MLP of the same hidden width by default.

    The ratios against the MLP only mean something at matched accuracy, which this
    does not check; pass --energy-baseline for an MLP that reaches the same accuracy.
    """
    config = model.config
    return report_from_args(model, args, default_baseline=(config.in_dim, config.n, config.num_classes))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the five controls against a checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--rk4-refine", type=int, default=10)
    parser.add_argument("--probe-epochs", type=int, default=40)
    parser.add_argument("--json-out", default=None, help="optional path for the raw numbers")
    parser.add_argument("--energy-only", action="store_true",
                        help="skip the controls; print only the energy estimate (needs no data)")
    add_energy_args(parser)
    args = parser.parse_args()

    device = pick_device(args.device)
    model, ckpt = load_checkpoint(args.checkpoint, device)
    hp = ckpt["hparams"]
    set_seed(hp["seed"])

    results: dict[str, float] = {}
    if not args.energy_only:
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
        print(format_controls(results, args.rk4_refine, model.config.num_steps))
        print()

    energy, table = run_energy(model, args)
    print(table)
    results.update(energy)

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()

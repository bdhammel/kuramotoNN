"""Train the coupling matrix. K is the only tensor that receives a gradient.

    python train.py --epochs 30

Calibration runs once before training and prints the init diagnostic table. The
five controls run once after training and are logged as wandb summary values.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
import wandb

from pymoto import calibrate, create_model
from pymoto.diagnostics import (
    assert_only_K_is_trainable,
    compute_diagnostics,
    format_diagnostics,
    warn_on_diagnostics,
)

from data import load_mnist
from eval import evaluate, format_controls, run_controls
from utils import pick_device, save_checkpoint, set_seed


@dataclass
class HParams:
    """Every knob. One dataclass, no config framework."""

    n: int = 256
    T: float = 1.0
    num_steps: int = 10
    g: float = 1.0
    k_scale: float = 1.0
    lr: float = 1e-3
    batch_size: int = 128
    epochs: int = 30
    val_size: int = 5000
    calibration_size: int = 1024
    seed: int = 0


def parse_args() -> tuple[HParams, argparse.Namespace]:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = HParams()
    parser.add_argument("--n", type=int, default=defaults.n)
    parser.add_argument("--num-steps", type=int, default=defaults.num_steps)
    parser.add_argument("--g", type=float, default=defaults.g,
                        help="drive gain in radians; usable range roughly [0.5, 1.5]")
    parser.add_argument("--k-scale", type=float, default=defaults.k_scale,
                        help="coupling scale; raise or lower if rho leaves [0.3, 2]")
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--wandb-project", default="kuramoto-mnist")
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--rk4-refine", type=int, default=10)
    parser.add_argument("--probe-epochs", type=int, default=40)
    parser.add_argument("--calibrate-only", action="store_true",
                        help="calibrate, print the init diagnostic table, and stop")
    args = parser.parse_args()

    hp = HParams(
        n=args.n,
        num_steps=args.num_steps,
        g=args.g,
        k_scale=args.k_scale,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
    )
    return hp, args


def main() -> None:
    hp, args = parse_args()
    device = pick_device(args.device)
    generator = set_seed(hp.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    run = wandb.init(
        project=args.wandb_project,
        mode=args.wandb_mode,
        config={**asdict(hp), "device": str(device)},
    )
    print(f"seed={hp.seed} device={device} run={run.name}")

    data = load_mnist(
        root=args.data_root,
        batch_size=hp.batch_size,
        val_size=hp.val_size,
        seed=hp.seed,
        calibration_size=hp.calibration_size,
    )
    x_cal = data.calibration.to(device)

    # Build on CPU with the seeded generator, then move: keeps init reproducible
    # regardless of which device the run lands on.
    model = create_model(
        "kuramoto_mnist",
        n=hp.n,
        T=hp.T,
        num_steps=hp.num_steps,
        k_scale=hp.k_scale,
        generator=generator,
    ).to(device)
    assert_only_K_is_trainable(model)
    coupling = model.get_coupling()

    calibrate(model, x_cal, g=hp.g)
    K_init = coupling.K.detach().cpu().clone()

    init_diag = compute_diagnostics(model, x_cal)
    calibrated = {
        "g (drive gain, radians)": float(model.kuramoto.dynamics.g),
        "tau (logit temperature)": float(model.get_classifier().tau),
        "k_scale": float(coupling.k_scale),
    }
    print(format_diagnostics(init_diag, "Init diagnostics (t = T, K at init)", calibrated))
    for warning in warn_on_diagnostics(init_diag):
        print(f"  WARNING: {warning}")

    run.summary.update(init_diag.as_dict(prefix="init/"))
    run.summary.update(
        {"init/g": calibrated["g (drive gain, radians)"],
         "init/tau": calibrated["tau (logit temperature)"],
         "init/k_scale": calibrated["k_scale"], "seed": hp.seed}
    )

    if args.calibrate_only:
        run.finish()
        return

    optimizer = torch.optim.Adam([coupling.K], lr=hp.lr)
    best_val_acc, global_step = 0.0, 0
    best_path = os.path.join(args.out_dir, "best")
    final_path = os.path.join(args.out_dir, "final")
    data_stats = {"mean": data.mean, "std": data.std}
    hp_dict = asdict(hp)

    for epoch in range(hp.epochs):
        model.train()
        running_loss, running_correct, seen = 0.0, 0, 0
        for x, y in data.train:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            global_step += 1
            running_loss += float(loss) * y.numel()
            running_correct += int((logits.argmax(dim=-1) == y).sum())
            seen += y.numel()
            wandb.log({"train/loss": float(loss)}, step=global_step)

        # Train metrics are accumulated during the pass rather than re-measured
        # afterwards: a second full sweep over 55k images per epoch would roughly
        # double runtime to sharpen a number that is only ever read as a trend.
        train_loss, train_acc = running_loss / seen, running_correct / seen
        val_loss, val_acc = evaluate(model, data.val, device)
        diag = compute_diagnostics(model, x_cal)

        wandb.log(
            {
                "epoch": epoch,
                "train/loss_epoch": train_loss,
                "train/acc": train_acc,
                "val/loss": val_loss,
                "val/acc": val_acc,
                **diag.as_dict(prefix="diag/"),
            },
            step=global_step,
        )
        print(
            f"epoch {epoch:3d}  train {train_loss:.4f}/{train_acc:.4f}  "
            f"val {val_loss:.4f}/{val_acc:.4f}  "
            f"||K||_2 {diag.k_spectral_norm:.3f}  h*lam {diag.h_lambda_max:.3f}  "
            f"PR {diag.participation_ratio:.1f}  asym {diag.k_asymmetry:.3f}"
        )
        # ||K||_2 is the step-size headroom monitor: K is trained, so the stiffness
        # of the ODE is a learned quantity and nothing in the loss stops it from
        # walking into a regime that Euler at this h cannot resolve.
        if diag.h_lambda_max > 0.5:
            print(f"  WARNING: h*lambda_max(J) = {diag.h_lambda_max:.3g}; Euler step unsafe")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(best_path, model, hp_dict, K_init, data_stats,
                            extra={"epoch": epoch, "val_acc": val_acc})

    save_checkpoint(final_path, model, hp_dict, K_init, data_stats,
                    extra={"epoch": hp.epochs - 1, "best_val_acc": best_val_acc})

    results = run_controls(
        model,
        K_init,
        data.train,
        data.test,
        device,
        rk4_refine=args.rk4_refine,
        probe_epochs=args.probe_epochs,
    )
    print()
    print(format_controls(results, args.rk4_refine, model.config.num_steps))
    run.summary.update(results)
    run.summary.update({"best_val_acc": best_val_acc})

    final_diag = compute_diagnostics(model, x_cal)
    run.summary.update(final_diag.as_dict(prefix="final/"))
    print()
    print(format_diagnostics(final_diag, "Final diagnostics (t = T, K trained)", calibrated))
    print(f"\ncheckpoints: {best_path}  {final_path}")
    run.finish()


if __name__ == "__main__":
    main()

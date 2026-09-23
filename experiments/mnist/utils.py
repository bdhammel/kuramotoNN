"""Seeding, device selection and checkpoint I/O.

The model, calibration and diagnostics live in pymoto; this file is only the
experiment's plumbing.
"""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch
from torch import Tensor

from pymoto import KuramotoForClassification, checkpoint_filter_fn, create_model

TRAINING_STATE_NAME = "training_state.pt"


def set_seed(seed: int) -> torch.Generator:
    """Seed python, numpy and torch. Returns a CPU generator for explicit use."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return torch.Generator().manual_seed(seed)


def pick_device(requested: str = "auto") -> torch.device:
    """Resolve 'auto' to cuda, then mps, then cpu."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(
    path: str,
    model: KuramotoForClassification,
    hparams: dict[str, Any],
    K_init: Tensor,
    data_stats: dict[str, float],
    extra: dict[str, Any] | None = None,
) -> None:
    """Write a checkpoint directory: the model's save_pretrained plus training_state.pt.

    K_init is stored alongside the trained state because the random-K control in
    eval.py needs the exact initial coupling matrix, not a freshly reseeded one.
    """
    model.save_pretrained(path)
    torch.save(
        {
            "K_init": K_init.detach().cpu().clone(),
            "hparams": hparams,
            "data_stats": data_stats,
            "extra": extra or {},
        },
        os.path.join(path, TRAINING_STATE_NAME),
    )


def load_checkpoint(
    path: str, device: torch.device
) -> tuple[KuramotoForClassification, dict[str, Any]]:
    """Rebuild the model from a checkpoint. Returns (model on device, training state).

    `path` is a save_checkpoint directory, or a pre-pymoto single-file .pt whose
    flat K/W/H state dict is remapped by checkpoint_filter_fn. Either way the
    training state has K_init, hparams, data_stats and extra.
    """
    if os.path.isdir(path):
        model = KuramotoForClassification.from_pretrained(path)
        state = torch.load(os.path.join(path, TRAINING_STATE_NAME), map_location="cpu", weights_only=False)
        return model.to(device), state

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    hp = ckpt["hparams"]
    model = create_model(
        "kuramoto_mnist",
        n=hp["n"],
        T=hp["T"],
        num_steps=hp["num_steps"],
        k_scale=hp["k_scale"],
    )
    model.load_state_dict(checkpoint_filter_fn(ckpt["state_dict"]))
    return model.to(device), ckpt

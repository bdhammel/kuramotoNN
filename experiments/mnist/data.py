"""MNIST loading, standardization and splitting for the Kuramoto classifier.

Everything downstream sees flattened float32 tensors of shape (B, 784). The whole
dataset fits comfortably in memory (60000 x 784 float32 ~= 188 MB), so the images
are decoded once into a TensorDataset rather than transformed per item.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

IMAGE_DIM: int = 784
NUM_CLASSES: int = 10


@dataclass(frozen=True)
class MNISTBundle:
    """Loaders plus the pieces calibration needs.

    Attributes:
        train / val / test: DataLoaders yielding ((B, 784) float32, (B,) int64).
        calibration: (n_cal, 784) float32, a fixed slice of the training split.
            Reused for every diagnostic so numbers are comparable across epochs.
        mean / std: the global scalars used to standardize, kept for the record.
    """

    train: DataLoader
    val: DataLoader
    test: DataLoader
    calibration: Tensor
    mean: float
    std: float


def standardize(x: Tensor, mean: float, std: float) -> Tensor:
    """(N, 784) in [0, 1] -> (N, 784) standardized.

    A single global scalar mean/std over the training split.

    Alternative considered and rejected: per-pixel mean/std with an epsilon floor.
    MNIST border pixels are identically 0 in every training image, so their
    per-pixel std is exactly 0. The epsilon floor keeps that from being a NaN, but
    it then multiplies those pixels' (pure-noise) test-time deviations by 1/eps,
    which swamps the drive z with directions that carry no class information. One
    global scalar has no such degenerate direction.
    """
    out = (x - mean) / std
    assert torch.isfinite(out).all(), "standardized data contains NaN or Inf"
    return out


def _flatten_images(raw: Tensor) -> Tensor:
    """(N, 28, 28) uint8 -> (N, 784) float32 in [0, 1]."""
    x = raw.to(torch.float32).div_(255.0).reshape(raw.shape[0], -1)
    assert x.shape[1] == IMAGE_DIM, f"expected {IMAGE_DIM} pixels, got {x.shape[1]}"
    return x


def load_mnist(
    root: str = "./data",
    batch_size: int = 128,
    val_size: int = 5000,
    seed: int = 0,
    calibration_size: int = 1024,
) -> MNISTBundle:
    """Download MNIST, hold out `val_size` training images, standardize, wrap in loaders.

    The standardization statistics come from the training split only, so the
    validation and test splits stay clean.

    `calibration_size` defaults to 1024 rather than the training batch size because
    the participation ratio of the 512-dimensional features is bounded above by
    (batch - 1); a batch of 128 would cap it at 127 and make a genuinely
    high-dimensional feature set indistinguishable from a merely adequate one.
    """
    train_raw = datasets.MNIST(root, train=True, download=True)
    test_raw = datasets.MNIST(root, train=False, download=True)

    x_all = _flatten_images(train_raw.data)
    y_all = train_raw.targets.clone().to(torch.int64)
    x_test = _flatten_images(test_raw.data)
    y_test = test_raw.targets.clone().to(torch.int64)

    split_gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(x_all.shape[0], generator=split_gen)
    n_train = x_all.shape[0] - val_size
    assert n_train > calibration_size, "training split smaller than the calibration batch"
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    mean = float(x_all[train_idx].mean())
    std = float(x_all[train_idx].std())

    x_train = standardize(x_all[train_idx], mean, std)
    x_val = standardize(x_all[val_idx], mean, std)
    x_test = standardize(x_test, mean, std)
    y_train, y_val = y_all[train_idx], y_all[val_idx]

    loader_gen = torch.Generator().manual_seed(seed + 1)
    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        generator=loader_gen,
        drop_last=False,
    )
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=512, shuffle=False)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=512, shuffle=False)

    return MNISTBundle(
        train=train_loader,
        val=val_loader,
        test=test_loader,
        calibration=x_train[:calibration_size].clone(),
        mean=mean,
        std=std,
    )

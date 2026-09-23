"""pymoto: a framework for experimenting with Kuramoto oscillator networks.

Organized like Hugging Face transformers and timm:

    pymoto.layers         reusable blocks (drive, coupling, dynamics, integrators, readout, head)
    pymoto.models         model families, each a config + modeling file, plus the registry
    pymoto.controls       evaluation variants: severed input, random K, solver transfer, probe
    pymoto.diagnostics    calibration diagnostics and the only-K-is-trainable guard

    model = pymoto.create_model("kuramoto_mnist", num_steps=50)
    pymoto.calibrate(model, x_cal, g=1.0)
    logits = model(x)
    model.save_pretrained("runs/final")
    model = pymoto.KuramotoForClassification.from_pretrained("runs/final")
"""

from pymoto.models import create_model, list_models, register_model
from pymoto.models.kuramoto import (
    KuramotoConfig,
    KuramotoForClassification,
    KuramotoModel,
    KuramotoModelOutput,
    calibrate,
    checkpoint_filter_fn,
)

__version__ = "0.1.0"

__all__ = [
    "KuramotoConfig",
    "KuramotoForClassification",
    "KuramotoModel",
    "KuramotoModelOutput",
    "calibrate",
    "checkpoint_filter_fn",
    "create_model",
    "list_models",
    "register_model",
]

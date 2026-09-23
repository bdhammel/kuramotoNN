"""Model registry and factory, in the style of timm.create_model.

Each model file ends with small entrypoint functions decorated with
@register_model. The function name is the model name, and its body pins the
variant's defaults:

    @register_model
    def kuramoto_mnist(**kwargs):
        model_args = dict(n=256, in_dim=784, num_classes=10)
        return _create_kuramoto(**dict(model_args, **kwargs))

    model = create_model("kuramoto_mnist", num_steps=50)   # any default overridable
"""

from __future__ import annotations

import fnmatch
from typing import Any, Callable

from torch import nn

_model_entrypoints: dict[str, Callable[..., nn.Module]] = {}


def register_model(fn: Callable[..., nn.Module]) -> Callable[..., nn.Module]:
    """Decorator: register `fn` under its own name."""
    name = fn.__name__
    if name in _model_entrypoints:
        raise ValueError(f"model {name!r} is already registered")
    _model_entrypoints[name] = fn
    return fn


def list_models(pattern: str = "*") -> list[str]:
    """Registered model names matching a shell-style `pattern`, sorted."""
    return sorted(fnmatch.filter(_model_entrypoints, pattern))


def create_model(model_name: str, **kwargs: Any) -> nn.Module:
    """Build a registered model. `kwargs` override the variant's defaults."""
    if model_name not in _model_entrypoints:
        raise ValueError(f"unknown model {model_name!r}; available: {list_models()}")
    return _model_entrypoints[model_name](**kwargs)

"""Model families. Importing a family registers its variants with create_model."""

from pymoto.models import kuramoto  # noqa: F401  (runs @register_model)
from pymoto.models._registry import create_model, list_models, register_model

__all__ = ["create_model", "list_models", "register_model"]

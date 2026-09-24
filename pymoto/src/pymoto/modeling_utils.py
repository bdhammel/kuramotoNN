"""PreTrainedModel: the base class every pymoto model inherits.

Mirrors Hugging Face's modeling_utils.PreTrainedModel, minus everything this project
does not need (hub downloads, sharding, dtype/device maps, weight tying):

    config_class        which PretrainedConfig subclass describes this model
    base_model_prefix   attribute name of the base model inside a task model, so
                        `.base_model` resolves to it on both (HF convention)
    save_pretrained     directory with config.json + pytorch_model.bin
    from_pretrained     rebuild from that directory: config first, then weights
    set_grad_checkpointing   memory-for-compute switch, forwarded to every block

A saved directory is self-describing -- the config carries the architecture and the
state dict carries everything learned or calibrated (K, W's scale, g, tau).
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Self

import torch
from torch import nn

from pymoto.configuration_utils import PretrainedConfig

WEIGHTS_NAME = "pytorch_model.bin"


class PreTrainedModel(nn.Module):
    """Base class for pymoto models. Subclasses set `config_class` and `base_model_prefix`."""

    config_class: ClassVar[type[PretrainedConfig]]
    base_model_prefix: ClassVar[str] = ""

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()
        if not isinstance(config, self.config_class):
            raise TypeError(
                f"{type(self).__name__} expects a {self.config_class.__name__}, got {type(config).__name__}"
            )
        self.config = config

    @property
    def base_model(self) -> nn.Module:
        """The base model: `self.<base_model_prefix>` on a task model, `self` on a base model."""
        return getattr(self, self.base_model_prefix, self)

    def set_grad_checkpointing(self, enable: bool = True) -> None:
        """Turn gradient checkpointing on or off in every block that supports it.

        timm's name for HF's gradient_checkpointing_enable(). A block opts in by
        having a `grad_checkpointing` attribute that its forward respects.
        """
        for module in self.modules():
            if hasattr(module, "grad_checkpointing"):
                module.grad_checkpointing = enable

    def save_pretrained(self, save_directory: str | Path) -> None:
        """Write config.json and pytorch_model.bin into `save_directory`."""
        save_directory = Path(save_directory)
        self.config.save_pretrained(save_directory)
        torch.save(self.state_dict(), save_directory / WEIGHTS_NAME)

    @classmethod
    def from_pretrained(cls, directory: str | Path, map_location: str | torch.device = "cpu") -> Self:
        """Rebuild from a save_pretrained directory. The state dict is loaded strictly."""
        directory = Path(directory)
        model = cls(cls.config_class.from_pretrained(directory))
        state_dict = torch.load(directory / WEIGHTS_NAME, map_location=map_location, weights_only=True)
        model.load_state_dict(state_dict)
        return model

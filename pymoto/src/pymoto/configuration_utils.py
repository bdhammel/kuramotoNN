"""PretrainedConfig: the serializable description of an architecture.

Mirrors Hugging Face's configuration_utils.PretrainedConfig. A config holds every
constructor argument a model needs, and nothing that is learned or calibrated --
those live in the state dict. Rebuilding a model is therefore always

    model = ModelClass(ConfigClass.from_pretrained(directory))

with no hand-maintained hparams dict in between.

Subclasses are plain dataclasses that set `model_type`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, ClassVar, Self

CONFIG_NAME = "config.json"


@dataclass
class PretrainedConfig:
    """Base class for model configs. Subclass as a dataclass and set `model_type`."""

    model_type: ClassVar[str] = ""

    def to_dict(self) -> dict[str, Any]:
        return {"model_type": self.model_type, **asdict(self)}

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> Self:
        """Strict inverse of to_dict: unknown keys are an error, not silently dropped."""
        config_dict = dict(config_dict)
        model_type = config_dict.pop("model_type", cls.model_type)
        if model_type != cls.model_type:
            raise ValueError(f"{cls.__name__} expects model_type {cls.model_type!r}, got {model_type!r}")
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(config_dict) - known)
        if unknown:
            raise ValueError(f"unknown {cls.__name__} fields: {unknown}")
        return cls(**config_dict)

    def save_pretrained(self, save_directory: str | Path) -> None:
        """Write config.json into `save_directory`, creating it if needed."""
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        with open(save_directory / CONFIG_NAME, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2, sort_keys=True)
            fh.write("\n")

    @classmethod
    def from_pretrained(cls, directory: str | Path) -> Self:
        with open(Path(directory) / CONFIG_NAME) as fh:
            return cls.from_dict(json.load(fh))

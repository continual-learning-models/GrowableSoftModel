"""ModelSpec: a declarative description of a SoftModel model to create.

Deliberately minimal — the factory is a tool that MANUFACTURES models; it
learns nothing about any business itself, and no human declares model
"types" or schemas here. A spec is just an identity (+ optional description
and held-out examples for the quality gate). Everything else — feature
space, output form (numeric vs categorical), output vocabulary, capacity —
is learned by the MODEL from the data it is taught.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .data import read_jsonl


@dataclass
class ModelSpec:
    model_id: str
    description: str = ""
    holdout: list[dict] = field(default_factory=list)   # [{"input","target"}, ...]

    @classmethod
    def from_files(cls, model_id: str, holdout_path: Optional[str] = None, **kw) -> "ModelSpec":
        holdout = read_jsonl(Path(holdout_path)) if holdout_path else []
        return cls(model_id=model_id, holdout=holdout, **kw)

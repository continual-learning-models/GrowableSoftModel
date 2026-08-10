"""Configuration for SoftModel.

Note: there is deliberately NO base/foundation model here. A SoftModel model is
a small specialized network trained from scratch; the calling LLM supplies all
general capability.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict, field
from pathlib import Path


@dataclass
class Config:
    # Backend: "mlp" (the real organ: numpy net, default) or
    #          "mock" (zero-dep text-lookup stub for control-loop demos/CI).
    backend: str = "mlp"

    # Where per-model data + versioned weights live.
    models_root: Path = (Path(__file__).resolve().parents[3]
                     / "trained_models" / "softmodel")

    # --- Organ (TinyMLP) training settings ---
    # hidden_sizes None = the model sizes its own capacity from its data
    # (soft size); set a tuple to pin it.
    hidden_sizes: tuple[int, ...] | None = field(default=None)
    epochs: int = 300
    lr: float = 1e-2
    batch_size: int = 32
    seed: int = 7

    # --- Discovery engine (Type-D models, M3) ---
    max_rule_conditions: int = 2     # up to pairwise interactions
    min_rule_support: int = 2        # a rule must cover at least N rows
    min_rule_confidence: float = 0.8

    # --- Eval gate ---
    min_gain: float = 1e-9   # candidate must exceed live metric by more than this

    # --- Drift awareness (M2) ---
    # Gate/drift checks score on the most recent N held-out examples
    # (None = all, i.e. M1-compatible behavior).
    gate_recent_n: int | None = None
    # Active version counts as drifted when its recent-slice metric falls more
    # than this below its score recorded at promotion time.
    drift_tolerance: float = 0.1
    # Default training window over the accumulated store (None = full replay).
    train_window_n: int | None = None

    @classmethod
    def from_env(cls, **overrides) -> "Config":
        cfg = cls()
        if os.getenv("SOFTMODEL_BACKEND"):
            cfg.backend = os.environ["SOFTMODEL_BACKEND"]
        if os.getenv("SOFTMODEL_MODELS_ROOT"):
            cfg.models_root = Path(os.environ["SOFTMODEL_MODELS_ROOT"])
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg

    def to_dict(self) -> dict:
        d = asdict(self)
        d["models_root"] = str(self.models_root)
        return d

"""Versioned model registry.

Each SoftModel model lives under models/<model_id>/ with:
  registry.json        -> lineage, active pointer, per-version scores, I/O schema
  weights/<version>/   -> that version's organ weights (+ its training store)
  eval/holdout.jsonl   -> held-out eval set  [{"input":..., "target":...}]

Version "v0" is the untrained organ (no weights). teach() creates v1, v2, ...
Only the active pointer and per-version weight dirs ever change.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

from .config import Config


class ModelRegistry:
    def __init__(self, config: Config):
        self.config = config
        self.root = Path(config.models_root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ---- paths ----
    def model_dir(self, model_id: str) -> Path:
        return self.root / model_id

    def reg_file(self, model_id: str) -> Path:
        return self.model_dir(model_id) / "registry.json"

    def weights_dir(self, model_id: str, version: str) -> Path:
        return self.model_dir(model_id) / "weights" / version

    def holdout_path(self, model_id: str) -> Path:
        return self.model_dir(model_id) / "eval" / "holdout.jsonl"

    # ---- lifecycle ----
    def exists(self, model_id: str) -> bool:
        return self.reg_file(model_id).exists()

    def list_models(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(d.name for d in self.root.iterdir()
                      if (d / "registry.json").exists())

    def create_model(self, model_id: str, description: str = "") -> dict:
        """Manufacture a new (untrained) model. No types, no schemas: the
        factory learns nothing about the business — the model will shape
        itself from the data it is taught."""
        if self.exists(model_id):
            return self.load(model_id)
        (self.model_dir(model_id) / "weights").mkdir(parents=True, exist_ok=True)
        (self.model_dir(model_id) / "eval").mkdir(parents=True, exist_ok=True)
        data = {
            "model_id": model_id,
            "description": description,
            "active": "v0",                          # v0 = untrained
            "versions": [
                {"version": "v0", "parent": None, "score": None,
                 "note": "untrained", "created": time.time()}
            ],
        }
        self.save(model_id, data)
        return data

    def load(self, model_id: str) -> dict:
        return json.loads(self.reg_file(model_id).read_text())

    def save(self, model_id: str, data: dict) -> None:
        self.reg_file(model_id).write_text(json.dumps(data, indent=2, ensure_ascii=False))

    # ---- versions ----
    def next_version(self, model_id: str) -> str:
        data = self.load(model_id)
        n = max(int(v["version"][1:]) for v in data["versions"])
        return f"v{n + 1}"

    def add_version(self, model_id: str, version: str, parent: str,
                    score: Optional[float] = None, note: str = "") -> None:
        data = self.load(model_id)
        data["versions"].append({
            "version": version, "parent": parent, "score": score,
            "note": note, "created": time.time(),
        })
        self.save(model_id, data)

    def record_score(self, model_id: str, version: str, score: float) -> None:
        data = self.load(model_id)
        for v in data["versions"]:
            if v["version"] == version:
                v["score"] = score
        self.save(model_id, data)

    def get_score(self, model_id: str, version: str) -> Optional[float]:
        for v in self.load(model_id)["versions"]:
            if v["version"] == version:
                return v["score"]
        return None

    def versions(self, model_id: str) -> list[dict[str, Any]]:
        return self.load(model_id)["versions"]

    # ---- drift status (M2) ----
    def record_drift(self, model_id: str, info: dict) -> None:
        data = self.load(model_id)
        data["drift"] = info
        self.save(model_id, data)

    # ---- active pointer ----
    def active(self, model_id: str) -> str:
        return self.load(model_id)["active"]

    def set_active(self, model_id: str, version: str) -> None:
        data = self.load(model_id)
        known = {v["version"] for v in data["versions"]}
        if version not in known:
            raise ValueError(f"unknown version {version} for model {model_id}")
        data["active"] = version
        self.save(model_id, data)

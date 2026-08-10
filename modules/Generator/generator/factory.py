"""SoftModelFactory: the high-level facade.

The factory is a TOOL that manufactures models. It learns nothing about any
business itself — business data flows only into a model's own versioned
store, and the MODEL learns from it (including its own shape: feature space,
output form, vocabulary, capacity). The factory provides manufacturing and
care: create, serve, evaluate, gate, version, rollback, drift checks.

    f = SoftModelFactory()
    f.create(ModelSpec("my_model", holdout=[...]))
    f.teach("my_model", [{"input": {...}, "target": ...}])
    f.infer("my_model", {"feature": 1.0})
"""
from __future__ import annotations

import json
from typing import Optional

from .config import Config
from .spec import ModelSpec
from .registry import ModelRegistry
from .model_manager import ModelManager
from .trainer import Trainer
from .evaluator import Evaluator
from .evolve import Evolve
from .data import write_jsonl, read_jsonl, stamp_examples


class SoftModelFactory:
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config.from_env()
        self.registry = ModelRegistry(self.config)
        self.model_manager = ModelManager(self.config, self.registry)
        self.trainer = Trainer(self.config, self.registry, self.model_manager)
        self.evaluator = Evaluator(self.config, self.model_manager, self.registry)
        self.evolve_ctl = Evolve(self.config, self.registry, self.model_manager,
                                 self.trainer, self.evaluator)

    # ---- lifecycle ----
    def create(self, spec: ModelSpec) -> dict:
        data = self.registry.create_model(spec.model_id, description=spec.description)
        if spec.holdout:
            write_jsonl(self.registry.holdout_path(spec.model_id), spec.holdout)
        return data

    def exists(self, model_id: str) -> bool:
        return self.registry.exists(model_id)

    def _learned_shape(self, model_id: str, version: str) -> Optional[dict]:
        """The shape the MODEL learned from its data (None if untrained)."""
        path = self.registry.weights_dir(model_id, version) / "shape.json"
        if version == "v0" or not path.exists():
            return None
        return json.loads(path.read_text())

    def list_models(self) -> list[dict]:
        """Fleet summary — how the brain discovers what models exist and what
        shape each one has learned for itself."""
        out = []
        for model_id in self.registry.list_models():
            data = self.registry.load(model_id)
            active = data["active"]
            summary = {
                "model_id": model_id,
                "description": data.get("description", ""),
                "active_version": active,
                "active_score": self.registry.get_score(model_id, active),
                "n_versions": len(data.get("versions", [])),
                "learned_shape": self._learned_shape(model_id, active),
            }
            drift = data.get("drift")
            if drift:
                summary["drift"] = {"drifted": drift.get("drifted"),
                                    "needs_reteach": drift.get("needs_reteach")}
            out.append(summary)
        return out

    # ---- use / teach ----
    def infer(self, model_id: str, input_, version: Optional[str] = None) -> dict:
        version = version or self.registry.active(model_id)
        result = self.model_manager.infer(model_id, version, input_)
        return {**result, "version": version}

    def teach(self, model_id: str, examples: list[dict], mode: str = "sft",
              window: Optional[int] = None,
              recent_n: Optional[int] = None) -> dict:
        return self.evolve_ctl.teach(model_id, examples, mode, window=window,
                                     recent_n=recent_n)

    # ---- drift awareness (M2) ----
    def add_holdout(self, model_id: str, examples: list[dict]) -> dict:
        """Append fresh labeled reality to the held-out stream (timestamped)."""
        path = self.registry.holdout_path(model_id)
        rows = read_jsonl(path) + stamp_examples(list(examples))
        write_jsonl(path, rows)
        return {"model_id": model_id, "holdout_size": len(rows),
                "added": len(examples)}

    def check_drift(self, model_id: str, recent_n: Optional[int] = None) -> dict:
        return self.evolve_ctl.check_drift(model_id, recent_n=recent_n)

    # ---- discoveries (readable regularities, any model) ----
    def discoveries(self, model_id: str, version: Optional[str] = None) -> dict:
        """The validated regularities a model has mined from its own store —
        readable knowledge for the brain. Available for any model whose data
        shaped a categorical output; numeric-shaped models report so."""
        version = version or self.registry.active(model_id)
        path = self.registry.weights_dir(model_id, version) / "rules.json"
        base = {"model_id": model_id, "version": version,
                "metric": self.registry.get_score(model_id, version)}
        if version == "v0" or not path.exists():
            shape = self._learned_shape(model_id, version)
            note = ("untrained" if shape is None else
                    "numeric-shaped data; rule mining not applicable yet")
            return {**base, "n_rules": 0, "regularities": [], "note": note}
        from .rules import RuleList
        rl = RuleList.load(path)
        return {**base, "n_rules": len(rl.rules), "regularities": rl.describe()}

    # ---- inspection / control ----
    def evaluate(self, model_id: str, version: Optional[str] = None,
                 recent_n: Optional[int] = None) -> dict:
        return self.evaluator.eval_version(
            model_id, version or self.registry.active(model_id), recent_n=recent_n)

    def versions(self, model_id: str) -> dict:
        return {"active": self.registry.active(model_id),
                "versions": self.registry.versions(model_id)}

    def rollback(self, model_id: str, to: str) -> dict:
        self.registry.set_active(model_id, to)
        return {"active": self.registry.active(model_id)}

    def card(self, model_id: str) -> dict:
        data = self.registry.load(model_id)
        data["holdout_size"] = len(read_jsonl(self.registry.holdout_path(model_id)))
        data["learned_shape"] = self._learned_shape(model_id, data["active"])
        return data

"""Model manager: load a SoftModel model at a specific version and run inference.

Backend-agnostic surface: `infer(model_id, version, input) -> {"output", "confidence"}`.

There are no human-declared model "types": each trained version carries the
shape the MODEL learned from its own data (`shape.json`) — a numeric head
(output is a number) or a categorical head over its own learned vocabulary
(output is a label + confidence). An untrained model (v0) honestly answers
`output: null, confidence: 0`.

- default backend: the numpy TinyMLP organ. NO base model, NO language
  ability — the calling LLM supplies understanding.
- mock: a zero-dependency text-lookup stub used to exercise the
  train -> eval -> gate -> version control loop in demos/CI.
"""
from __future__ import annotations

import json
from typing import Any

from .config import Config
from .data import normalize, featurize
from .registry import ModelRegistry


class ModelManager:
    def __init__(self, config: Config, registry: ModelRegistry):
        self.config = config
        self.registry = registry
        self._cache: dict[tuple, Any] = {}

    # ---------- public ----------
    def infer(self, model_id: str, version: str, input_: Any) -> dict:
        if self.config.backend == "mock":
            return self._mock_infer(model_id, version, input_)
        return self._organ_infer(model_id, version, input_)

    def invalidate_cache(self, model_id: str) -> None:
        self._cache = {k: v for k, v in self._cache.items() if k[0] != model_id}

    # ---------- the organ (shape learned from data) ----------
    def _load(self, model_id: str, version: str):
        key = (model_id, version)
        if key in self._cache:
            return self._cache[key]
        from .nets import TinyMLP
        wdir = self.registry.weights_dir(model_id, version)
        loaded = None
        if (wdir / "organ.npz").exists() and (wdir / "shape.json").exists():
            rules = None
            if (wdir / "rules.json").exists():
                from .rules import RuleList
                rules = RuleList.load(wdir / "rules.json")
            loaded = (TinyMLP.load(wdir),
                      json.loads((wdir / "shape.json").read_text()),
                      rules)
        self._cache[key] = loaded
        return loaded

    def _organ_infer(self, model_id: str, version: str, input_: Any) -> dict:
        import numpy as np
        loaded = None if version == "v0" else self._load(model_id, version)
        if loaded is None:
            return {"output": None, "confidence": 0.0, "note": "untrained"}
        net, shape, rules = loaded
        x = np.array([featurize(input_, shape["features"])])
        if shape["mode"] == "numeric":
            value = float(net.predict_value(x)[0])
            if shape.get("integer"):
                value = int(round(value))
            return {"output": value, "confidence": None}
        probs = net.predict_proba(x)[0]
        idx = int(probs.argmax())
        result = {"output": shape["vocab"][idx],
                  "confidence": round(float(probs[idx]), 4)}
        # Explainability: if the model's own mined regularities agree, cite one.
        if rules is not None:
            cited = rules.predict(dict(zip(shape["features"], x[0].tolist())))
            if cited["output"] == result["output"] and cited.get("rule"):
                result["rule"] = cited["rule"]
        return result

    def predict_dist(self, model_id: str, version: str,
                     input_: Any) -> dict:
        """GSM-I1: the distribution infer() truncates — numeric
        point or the FULL categorical probability vector."""
        import numpy as np
        loaded = (None if version == "v0"
                  else self._load(model_id, version))
        if loaded is None:
            return {"kind": "none", "note": "untrained",
                    "version": version}
        net, shape, _ = loaded
        x = np.array([featurize(input_, shape["features"])])
        if shape["mode"] == "numeric":
            return {"kind": "numeric",
                    "value": float(net.predict_value(x)[0]),
                    "version": version}
        probs = net.predict_proba(x)[0]
        return {"kind": "categorical",
                "labels": list(shape["vocab"]),
                "probs": [float(v) for v in probs],
                "version": version}

    # ---------- mock backend (control-loop stub) ----------
    def _mock_map(self, model_id: str, version: str) -> dict:
        if version == "v0":
            return {}
        f = self.registry.weights_dir(model_id, version) / "memory.json"
        if not f.exists():
            return {}
        return json.loads(f.read_text()).get("map", {})

    def _mock_infer(self, model_id: str, version: str, input_: Any) -> dict:
        mapping = self._mock_map(model_id, version)
        key = normalize(str(input_))
        if key in mapping:
            return {"output": mapping[key], "confidence": 1.0}
        # Untrained behavior: echo (deliberately weak).
        return {"output": str(input_).strip(), "confidence": 0.0}

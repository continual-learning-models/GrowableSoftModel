"""Trainer: the MODEL's own learning process (the factory only invokes it).

`train_candidate(model_id, examples, parent_version, window) -> new_version`

A neural model has no human-declared "type". When taught, the model shapes
ITSELF from the data in its versioned store:

- **feature space** = the union of feature keys observed in its store;
- **output form**   = numeric head if every target parses as a number,
  otherwise a categorical head over the target values it has seen (its own
  output vocabulary — which may grow in later versions as new values appear);
- **capacity**      = hidden width auto-sized from its data (soft size),
  unless the config pins it;
- alongside the net, the model also mines its store for **readable
  regularities** (rules.json) when its targets are categorical — every model
  can answer `discoveries`, not a special "type" of model.

Evolution semantics (unchanged): each version carries its OWN accumulated
training store (parent's store + the new examples) and is retrained from
scratch on it — full replay kills catastrophic forgetting; a rejected
candidate's examples never pollute the live lineage. `window=N` trains on
only the most recent N stored examples (drift adaptation, M2).

The mock backend remains a zero-dep control-loop stub.
"""
from __future__ import annotations

import json
from typing import Optional

from .config import Config
from .data import normalize, featurize, read_jsonl, write_jsonl, recent_slice
from .registry import ModelRegistry
from .model_manager import ModelManager


def _is_number(s) -> bool:
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


def infer_shape(rows: list[dict]) -> dict:
    """The model reads its own store and decides its shape (no human types).

    Returns {"features": [...], "mode": "numeric"|"categorical",
             "vocab": [...] (categorical only), "integer": bool (numeric only)}
    """
    features: list[str] = []
    seen = set()
    for r in rows:
        if isinstance(r["input"], dict):
            for k in r["input"]:
                if k not in seen:
                    seen.add(k)
                    features.append(k)
    targets = [str(r["target"]) for r in rows]
    if targets and all(_is_number(t) for t in targets):
        values = [float(t) for t in targets]
        return {"features": features, "mode": "numeric",
                "integer": all(v == int(v) for v in values)}
    vocab = sorted(set(targets))
    return {"features": features, "mode": "categorical", "vocab": vocab}


def auto_hidden(n_in: int, n_out: int, n_rows: int) -> tuple[int, ...]:
    """Soft capacity: the model sizes itself from its data (floor 16 — below
    that, numeric mappings underfit; see the defaults sweep in the repo log)."""
    width = max(16, min(64, 4 * (n_in + n_out)))
    return (int(width),)


class Trainer:
    def __init__(self, config: Config, registry: ModelRegistry, model_manager: ModelManager):
        self.config = config
        self.registry = registry
        self.mm = model_manager

    def train_candidate(self, model_id: str, examples: list[dict],
                        parent_version: Optional[str] = None,
                        window: Optional[int] = None) -> str:
        parent = parent_version or self.registry.active(model_id)
        version = self.registry.next_version(model_id)
        if self.config.backend == "mock":
            self._train_mock(model_id, examples, parent, version)
        else:
            self._train_organ(model_id, examples, parent, version, window=window)
        note = f"taught {len(examples)} examples"
        if window:
            note += f" (window={window})"
        self.registry.add_version(model_id, version, parent=parent, note=note)
        return version

    # ---------- the model learns (and shapes itself) ----------
    def _train_organ(self, model_id, examples, parent, version, window=None):
        import numpy as np
        from .nets import TinyMLP
        from .rules import induce_rules

        store = read_jsonl(self._store_path(model_id, parent)) + list(examples)
        train_rows = recent_slice(store, window)
        if not train_rows:
            raise ValueError("no examples to learn from")

        shape = infer_shape(train_rows)
        features = shape["features"]
        if not features:
            raise ValueError("examples carry no feature keys to learn from")

        X = np.array([featurize(ex["input"], features) for ex in train_rows])
        hidden = (tuple(self.config.hidden_sizes)
                  if self.config.hidden_sizes else None)

        if shape["mode"] == "numeric":
            y = np.array([float(ex["target"]) for ex in train_rows])
            net = TinyMLP(len(features), hidden or auto_hidden(len(features), 1, len(train_rows)),
                          1, mode="numeric", seed=self.config.seed)
            net.fit(X, y, epochs=self.config.epochs, lr=self.config.lr,
                    batch_size=self.config.batch_size, seed=self.config.seed)
        else:
            vocab = shape["vocab"]
            y = np.array([vocab.index(str(ex["target"])) for ex in train_rows])
            net = TinyMLP(len(features), hidden or auto_hidden(len(features), len(vocab), len(train_rows)),
                          len(vocab), mode="categorical", seed=self.config.seed)
            net.fit(X, y, epochs=self.config.epochs, lr=self.config.lr,
                    batch_size=self.config.batch_size, seed=self.config.seed)

        out_dir = self.registry.weights_dir(model_id, version)
        net.save(out_dir)
        (out_dir / "shape.json").write_text(json.dumps(shape))

        # Readable regularities are mined for every categorically-shaped model
        # (not a special model type). Numeric stores skip rule mining for now.
        if shape["mode"] == "categorical":
            rule_list = induce_rules(
                train_rows, features, shape["vocab"],
                max_conditions=self.config.max_rule_conditions,
                min_support=self.config.min_rule_support,
                min_confidence=self.config.min_rule_confidence,
                max_rules=getattr(self.config, "max_rules", 32))
            rule_list.save(out_dir / "rules.json")

        write_jsonl(self._store_path(model_id, version), store)
        self.mm.invalidate_cache(model_id)

    def _store_path(self, model_id: str, version: str):
        return self.registry.weights_dir(model_id, version) / "train_store.jsonl"

    # ---------- mock (control-loop stub) ----------
    def _train_mock(self, model_id, examples, parent, version):
        mapping = dict(self.mm._mock_map(model_id, parent))  # start from parent
        for ex in examples:
            mapping[normalize(str(ex["input"]))] = ex["target"]
        out_dir = self.registry.weights_dir(model_id, version)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "memory.json").write_text(
            json.dumps({"map": mapping, "trained_on": len(examples)},
                       ensure_ascii=False, indent=2)
        )

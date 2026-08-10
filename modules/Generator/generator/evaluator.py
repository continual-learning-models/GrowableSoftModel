"""Evaluator + gate.

The Reality-Grounding gate: evaluate a version on the held-out set with an
automatic metric (label exact match / accuracy) and decide whether a candidate
may be promoted. Never uses an LLM as the judge (ChemCrow lesson) — the metric
is objective.
"""
from __future__ import annotations

from typing import Optional

from .config import Config
from .data import read_jsonl, recent_slice
from .registry import ModelRegistry
from .model_manager import ModelManager


class Evaluator:
    def __init__(self, config: Config, model_manager: ModelManager, registry: ModelRegistry):
        self.config = config
        self.mm = model_manager
        self.registry = registry
        # param-interface batch S3 (docs/system/22 item 6): the
        # factory gate's numeric tolerance — one rule with the
        # lifecycle gate; config key gate_tol, default 0.5
        t = getattr(config, "gate_tol", 0.5)
        if isinstance(t, bool) or not isinstance(t, (int, float)) \
                or t < 0:
            raise ValueError(
                f"gate_tol must be a number >= 0; got {t!r}")
        self.tol = float(t)

    def eval_version(self, model_id: str, version: str,
                     recent_n: Optional[int] = None) -> dict:
        """Score a version on the held-out set — optionally only on the most
        recent `recent_n` examples (drift-awareness, M2): promotion must be
        judged against reality as it is now."""
        holdout = read_jsonl(self.registry.holdout_path(model_id))
        rows = recent_slice(holdout, recent_n)
        if not rows:
            return {"metric": 0.0, "n": 0, "correct": 0, "recent_n": recent_n}
        correct = 0
        for row in rows:
            pred = self.mm.infer(model_id, version, row["input"])["output"]
            if self._match(pred, row["target"], tol=self.tol):
                correct += 1
        return {"metric": correct / len(rows), "n": len(rows), "correct": correct,
                "recent_n": recent_n}

    @staticmethod
    def _match(pred, target, tol=0.5) -> bool:
        """Data-driven comparison — no declared output types. If both sides
        are numbers, a numeric prediction counts as correct within
        +-tol (default 0.5; exact for integer-valued data);
        otherwise exact string match."""
        if pred is None:
            return False
        try:
            return abs(float(pred) - float(target)) <= tol
        except (TypeError, ValueError):
            return str(pred).strip() == str(target).strip()

    def gate(self, cand: dict, live: Optional[dict]) -> bool:
        """Promote only if the candidate strictly beats the live version."""
        if live is None:
            return True
        return cand["metric"] > live["metric"] + self.config.min_gain

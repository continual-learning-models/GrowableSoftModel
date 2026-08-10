"""Rule-discovery engine for Type-D SoftModel models (M3).

Deterministic decision-list induction (separate-and-conquer): mines ordered
`IF cond [AND cond] THEN class` regularities from observations. The induced
rule set is the model's capability — validated on held-out data by the same
Reality-Grounding gate as every other SoftModel model, and returned to the
brain (LLM) as human-readable statements via `discoveries`.

Pure Python, no dependencies. Hypothesis space is intentionally the simplest
honest one: per-feature thresholds and binary equalities, with up to
`max_conditions` (default 2) conjunctive conditions — enough to surface
threshold effects and pairwise interactions from raw data.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

_OPS = {
    ">=": lambda v, t: v >= t,
    "<=": lambda v, t: v <= t,
    "==": lambda v, t: v == t,
}

_MAX_RULES = 32  # safety cap


@dataclass(frozen=True)
class Condition:
    feature: str
    op: str
    threshold: float

    def match(self, x: dict) -> bool:
        return _OPS[self.op](float(x.get(self.feature, 0.0)), self.threshold)

    def describe(self) -> str:
        t = self.threshold
        t_str = str(int(t)) if float(t) == int(t) else f"{t:.4g}"
        return f"{self.feature} {self.op} {t_str}"


@dataclass
class Rule:
    conditions: tuple[Condition, ...]
    target: str
    confidence: float
    support: int

    def match(self, x: dict) -> bool:
        return all(c.match(x) for c in self.conditions)

    def describe(self) -> str:
        cond = " AND ".join(c.describe() for c in self.conditions)
        return (f"IF {cond} THEN {self.target}"
                f"  (confidence {self.confidence:.2f}, support {self.support})")


class RuleList:
    """An ordered rule list + default class: the discovered regularities."""

    def __init__(self, rules: list[Rule], default: str, default_confidence: float,
                 features: list[str], classes: list[str]):
        self.rules = rules
        self.default = default
        self.default_confidence = default_confidence
        self.features = features
        self.classes = classes

    # ---------- inference (explainable) ----------
    def predict(self, x: dict) -> dict:
        for i, rule in enumerate(self.rules):
            if rule.match(x):
                return {"output": rule.target,
                        "confidence": round(rule.confidence, 4),
                        "rule": rule.describe(), "rule_index": i}
        return {"output": self.default,
                "confidence": round(self.default_confidence, 4),
                "rule": f"DEFAULT {self.default}", "rule_index": None}

    def describe(self) -> list[str]:
        lines = [r.describe() for r in self.rules]
        lines.append(f"OTHERWISE {self.default}"
                     f"  (confidence {self.default_confidence:.2f})")
        return lines

    # ---------- persistence ----------
    def to_dict(self) -> dict:
        return {
            "rules": [{"conditions": [{"feature": c.feature, "op": c.op,
                                       "threshold": c.threshold}
                                      for c in r.conditions],
                       "target": r.target, "confidence": r.confidence,
                       "support": r.support} for r in self.rules],
            "default": self.default,
            "default_confidence": self.default_confidence,
            "features": self.features,
            "classes": self.classes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RuleList":
        rules = [Rule(tuple(Condition(c["feature"], c["op"], c["threshold"])
                            for c in r["conditions"]),
                      r["target"], r["confidence"], r["support"])
                 for r in d["rules"]]
        return cls(rules, d["default"], d["default_confidence"],
                   d["features"], d["classes"])

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> "RuleList":
        return cls.from_dict(json.loads(Path(path).read_text()))


# ---------------------------------------------------------------- induction

def _candidate_conditions(rows: list[dict], features: list[str],
                          max_thresholds: int = 16) -> list[Condition]:
    conds: list[Condition] = []
    for f in features:
        vals = sorted({float(r["input"].get(f, 0.0)) for r in rows})
        if len(vals) <= 1:
            continue
        if set(vals) <= {0.0, 1.0}:   # binary feature
            conds.append(Condition(f, "==", 1.0))
            conds.append(Condition(f, "==", 0.0))
            continue
        mids = [(a + b) / 2.0 for a, b in zip(vals, vals[1:])]
        if len(mids) > max_thresholds:
            step = len(mids) / max_thresholds
            mids = [mids[int(i * step)] for i in range(max_thresholds)]
        for m in mids:
            conds.append(Condition(f, ">=", m))
            conds.append(Condition(f, "<=", m))
    return conds


def _evaluate(conditions: tuple[Condition, ...], target: str,
              rows: list[dict]) -> Optional[tuple]:
    matched = [r for r in rows if all(c.match(r["input"]) for c in conditions)]
    if not matched:
        return None
    hits = sum(1 for r in matched if str(r["target"]) == target)
    confidence = hits / len(matched)
    # Selection key: confidence, then support, then fewer conditions, then
    # prefer categorical (==) conditions — they are scale-robust and more
    # interpretable than sample-dependent midpoints, so at equal quality the
    # crisper regularity (e.g. an interaction of binary factors) surfaces.
    n_eq = sum(1 for c in conditions if c.op == "==")
    return (confidence, len(matched), -len(conditions), n_eq, conditions, target)


def induce_rules(rows: list[dict], features: list[str], classes: list[str],
                 max_conditions: int = 2, min_support: int = 2,
                 min_confidence: float = 0.8, max_thresholds: int = 16,
                 pair_seed_pool: int = 32,
                 max_rules: int = _MAX_RULES) -> RuleList:
    """Mine an ordered rule list from observations [{"input": {...}, "target": s}].

    Deterministic separate-and-conquer: repeatedly select the best rule on the
    not-yet-covered rows (max confidence, then support, then simplicity; must
    meet min_confidence/min_support), remove the rows it covers, repeat.
    """
    conds = _candidate_conditions(rows, features, max_thresholds)
    remaining = list(rows)
    rules: list[Rule] = []

    while len(remaining) >= min_support and len(rules) < max_rules and conds:
        candidates: list[tuple] = []
        for target in classes:
            singles: list[tuple] = []
            for c in conds:
                ev = _evaluate((c,), target, remaining)
                if ev is not None:
                    singles.append(ev)
            candidates.extend(singles)
            if max_conditions >= 2 and singles:
                # extend the strongest singles with a condition on another feature
                seeds = sorted(singles, key=lambda s: (-s[0], -s[1]))[:pair_seed_pool]
                for ev_single in seeds:
                    seed = ev_single[4][0]
                    for c2 in conds:
                        if c2.feature == seed.feature:
                            continue
                        ev = _evaluate((seed, c2), target, remaining)
                        if ev is not None:
                            candidates.append(ev)
        passing = [c for c in candidates
                   if c[0] >= min_confidence and c[1] >= min_support]
        if not passing:
            break
        best = max(passing, key=lambda s: (s[0], s[1], s[2], s[3]))
        confidence, support, _, _, conditions, target = best
        rules.append(Rule(tuple(conditions), target, confidence, support))
        remaining = [r for r in remaining
                     if not all(c.match(r["input"]) for c in conditions)]

    base = remaining if remaining else rows
    counts = {cls: sum(1 for r in base if str(r["target"]) == cls)
              for cls in classes}
    default = max(classes, key=lambda cls: counts[cls])   # ties -> class order
    default_confidence = counts[default] / len(base) if base else 0.0
    return RuleList(rules, default, default_confidence, list(features), list(classes))

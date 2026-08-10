"""Data helpers: jsonl IO, text normalization (mock), and featurization (mlp).

The brain-organ I/O contract: the calling LLM extracts
structured features; the organ consumes a schema-ordered numeric vector.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Iterable


_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")


def normalize(text: str) -> str:
    """Normalization key used by the mock backend (lowercase, strip
    punctuation, collapse whitespace)."""
    t = _PUNCT.sub(" ", str(text).lower())
    return _WS.sub(" ", t).strip()


def read_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))


def stamp_examples(rows: list[dict]) -> list[dict]:
    """Ensure every example carries a timestamp `t` (drift-awareness, M2).

    Examples that already have `t` keep it; new ones get the current time,
    with microsecond increments preserving their relative order.
    """
    now = time.time()
    return [{**r, "t": r.get("t", now + i * 1e-6)} for i, r in enumerate(rows)]


def recent_slice(rows: list[dict], recent_n: int | None) -> list[dict]:
    """The most recent `recent_n` rows by timestamp (stable for untimestamped
    rows, which sort first in file order). None returns all rows."""
    ordered = sorted(rows, key=lambda r: r.get("t", 0.0))
    if recent_n is None or recent_n <= 0 or recent_n >= len(ordered):
        return ordered
    return ordered[-recent_n:]


def featurize(example_input: Any, feature_names: list[str]) -> list[float]:
    """Turn an organ input into a schema-ordered numeric vector.

    Accepts:
      - dict  {feature_name: number}  (missing features default to 0.0)
      - list/tuple of numbers         (must match the schema length)
    """
    if isinstance(example_input, dict):
        return [float(example_input.get(name, 0.0)) for name in feature_names]
    if isinstance(example_input, (list, tuple)):
        if len(example_input) != len(feature_names):
            raise ValueError(
                f"vector length {len(example_input)} != schema length {len(feature_names)}")
        return [float(v) for v in example_input]
    raise ValueError(
        "organ input must be a feature dict or a numeric vector "
        f"(got {type(example_input).__name__}); the calling LLM should extract "
        "features per the model's input_features schema")

"""Experience store (Stage 2): the model's episodic memory.

study() rows are retained here (reservoir-sampled up to a cap) so
self-reprocessing (consolidation replay, Phi re-founding sizing and
retraining, variation probes) has real data to chew on.

QUARANTINE INVARIANT (non-negotiable, theory Part A):
the holdout is the commit gate's reality standard. Rows that enter any
holdout stream are registered here by content hash and are REFUSED by
the store, loudly — self-reprocessing must never train on the gate's
exam. Tests enforce both directions (source tag and content hash).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def row_hash(row) -> str:
    return hashlib.sha1(
        json.dumps(row, sort_keys=True, default=str).encode()).hexdigest()


class QuarantineViolation(ValueError):
    """Raised loudly on any attempt to store holdout data."""


class ExperienceStore:
    """Reservoir-sampled, provenance-tagged, holdout-quarantined JSONL
    store. Deterministic under (seed, insertion order)."""

    def __init__(self, dir_path, cap: int = 5000, seed: int = 7):
        self.dir = Path(dir_path)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.cap = int(cap)
        self._rng = np.random.default_rng(seed)
        self._seed = seed
        self.rows: list = []          # [{"row":..., "source":...}]
        self.n_seen = 0               # total offered (for reservoir)
        self.n_refused = 0
        self._quarantine: set = set()
        self._load()

    # ---------------- persistence ----------------
    def _state_path(self):
        return self.dir / "store_state.json"

    def _rows_path(self):
        return self.dir / "experience.jsonl"

    def _load(self):
        if self._state_path().exists():
            st = json.loads(self._state_path().read_text())
            self.n_seen = st["n_seen"]
            self.n_refused = st.get("n_refused", 0)
            self._quarantine = set(st["quarantine"])
            self._rng = np.random.default_rng(st["rng_seed_next"])
        if self._rows_path().exists():
            self.rows = [json.loads(l)
                         for l in self._rows_path().read_text().splitlines()]

    def save(self):
        with open(self._rows_path(), "w") as fh:
            for r in self.rows:
                fh.write(json.dumps(r, default=str) + "\n")
        # derive the next rng seed deterministically from consumption
        self._state_path().write_text(json.dumps(
            {"n_seen": self.n_seen, "n_refused": self.n_refused,
             "quarantine": sorted(self._quarantine),
             "rng_seed_next": int(self._seed + self.n_seen)}))

    # ---------------- quarantine ----------------
    def register_holdout(self, rows):
        """Called whenever rows enter a holdout stream. From now on the
        store refuses these contents."""
        for r in rows:
            self._quarantine.add(row_hash(r))
        # holdout rows already stored by an earlier add() are evicted
        before = len(self.rows)
        self.rows = [e for e in self.rows
                     if e["hash"] not in self._quarantine]
        return {"registered": len(rows), "evicted": before - len(self.rows)}

    # ---------------- ingestion (reservoir) ----------------
    def add(self, rows, source: str):
        """Offer rows to the reservoir. source is a provenance tag;
        'holdout' (or registered holdout content) is refused LOUDLY."""
        if source == "holdout":
            raise QuarantineViolation(
                "experience store refuses holdout-sourced rows")
        added = 0
        for r in rows:
            h = row_hash(r)
            if h in self._quarantine:
                self.n_refused += 1
                raise QuarantineViolation(
                    "row content is registered holdout reality — "
                    "refusing to store the gate's exam")
            self.n_seen += 1
            entry = {"row": r, "source": source, "hash": h,
                     "t": self.n_seen}
            if len(self.rows) < self.cap:
                self.rows.append(entry)
                added += 1
            else:                     # reservoir replacement
                j = int(self._rng.integers(0, self.n_seen))
                if j < self.cap:
                    self.rows[j] = entry
                    added += 1
        return {"offered": len(rows), "kept_or_replaced": added,
                "size": len(self.rows)}

    # ---------------- consumption ----------------
    def sample(self, n, rng=None):
        rng = rng or self._rng
        n = min(int(n), len(self.rows))
        idx = rng.choice(len(self.rows), size=n, replace=False)
        return [self.rows[i]["row"] for i in idx]

    def all_rows(self):
        return [e["row"] for e in self.rows]

    def __len__(self):
        return len(self.rows)

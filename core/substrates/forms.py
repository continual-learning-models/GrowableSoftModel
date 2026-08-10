"""Data-form detection (SUBSTRATE_ARCHITECTURE Section 3; plan S3.1).

Self-shaping extended to the FORM of the data: the system detects which
input form the examples carry, so substrate compatibility can be checked
and the AI's substrate choice can be informed. Frozen modules untouched.
"""
from __future__ import annotations

import numpy as np

FORMS = ("vector", "sequence", "grid", "graph")


def detect_form(rows) -> str | None:
    """Detect the data form from example rows [{input, target}, ...].
    Returns one of FORMS, or None if malformed/mixed."""
    if not rows:
        return None
    forms = {_form_of(r.get("input")) for r in rows[:32]}
    forms.discard(None)
    return forms.pop() if len(forms) == 1 else None


def _form_of(x):
    if isinstance(x, dict):
        if {"nodes", "edges"} <= set(x):
            return "graph"
        if all(np.isscalar(v) or isinstance(v, (int, float))
               for v in x.values()):
            return "vector"
        return None
    if isinstance(x, (list, tuple, np.ndarray)):
        arr = np.asarray(x, dtype=object)
        try:
            arr_f = np.asarray(x, dtype=float)
        except (ValueError, TypeError):
            return None
        if arr_f.ndim == 1:
            return "vector"                     # plain numeric array
        if arr_f.ndim == 2:
            # ordered steps of feature vectors -> sequence;
            # (grid detection: square-ish 2D with many rows AND cols)
            r, c = arr_f.shape
            if r >= 8 and c >= 8:
                return "grid"
            return "sequence"
        return None
    return None


# default substrate per detected form (transparent auto-default, S3.2)
FORM_DEFAULT = {"vector": "mlp", "sequence": "sequence",
                "grid": None, "graph": None}   # None -> refusal until registered


def interaction_probe(rows, seed=0) -> float:
    """Advisory relation signal for vector data: how much do pairwise
    feature PRODUCTS explain the target beyond linear terms alone?
    Returns gain in [0, 1]; high gain -> relation-heavy -> transformer."""
    import itertools
    feats = list(rows[0]["input"].keys())
    X = np.array([[float(r["input"][k]) for k in feats] for r in rows])
    try:
        y = np.array([float(r["target"]) for r in rows])
    except (ValueError, TypeError):
        vocab = sorted({str(r["target"]) for r in rows})
        y = np.array([vocab.index(str(r["target"])) for r in rows], float)
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    ys = (y - y.mean()) / (y.std() + 1e-9)

    def r2(D):
        D1 = np.hstack([D, np.ones((len(D), 1))])
        w, *_ = np.linalg.lstsq(D1, ys, rcond=None)
        resid = ys - D1 @ w
        return 1.0 - float(np.mean(resid ** 2))

    lin = r2(Xs)
    pairs = [Xs[:, i] * Xs[:, j]
             for i, j in itertools.combinations(range(Xs.shape[1]), 2)]
    inter = r2(np.column_stack([Xs] + pairs)) if pairs else lin
    return max(0.0, min(1.0, inter - lin))
